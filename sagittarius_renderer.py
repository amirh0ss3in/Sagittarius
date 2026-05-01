import os
import math
import hashlib
import json
import numpy as np
import cv2
import warp as wp

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    _rich_available = True
    console = Console()
except ImportError:
    _rich_available = False
    class _DummyConsole:
        def print(self, text): print(text)
    console = _DummyConsole()

try:
    import psutil
    _psutil_available = True
except ImportError:
    _psutil_available = False

wp.init()

# ---------------------------------------------------------
# WARP STRUCTS FOR CONFIGURATION
# ---------------------------------------------------------

@wp.struct
class SimConfig:
    cylinder_min_radius: float
    cylinder_max_radius: float
    cylinder_half_height: float
    dr: float
    dtheta: float
    dy: float
    grid_r: int
    grid_theta: int
    grid_y: int
    dt_sim: float
    advection_strength: float
    dissipation: float
    orbital_velocity_scale: float
    gravity_strength: float
    orbital_assist_strength: float
    max_velocity_physical: float
    gm: float
    vertical_noise_scale: float
    vertical_strength: float
    warp_field_scale: float
    warp_strength: float
    tangential_stretch: float
    filament_noise_scale: float
    filament_contrast: float
    clump_noise_scale: float
    clump_strength: float
    disk_noise_strength: float

@wp.struct
class RenderConfig:
    gm: float
    rs: float
    horizon_radius: float
    disk_inner: float
    disk_outer: float
    disk_half_thickness: float
    far_field: float
    volume_substeps: int
    density_multiplier: float
    density_pow: float
    emission_strength: float
    absorption_coefficient: float
    scattering_strength: float
    hg_asymmetry: float
    self_shadow_strength: float
    doppler_strength: float
    eq_shadow_width: float
    eq_shadow_strength: float
    dt_initial: float
    dt_min: float
    dt_max: float
    max_steps: int
    tolerance: float

# ---------------------------------------------------------
# MATH & NOISE HELPERS
# ---------------------------------------------------------

@wp.func
def fract(x: float):
    return x - wp.floor(x)

@wp.func
def fract_vec3(v: wp.vec3):
    return wp.vec3(fract(v[0]), fract(v[1]), fract(v[2]))

@wp.func
def hash31(p: wp.vec3):
    p3 = fract_vec3(wp.cw_mul(p, wp.vec3(437.585453, 223.13306, 353.72935)))
    dot_val = wp.dot(p3, p3 + wp.vec3(19.19, 19.19, 19.19))
    p3 = p3 + wp.vec3(dot_val, dot_val, dot_val)
    return fract((p3[0] + p3[1]) * p3[2])

@wp.func
def value_noise_3d(p: wp.vec3):
    i = wp.vec3(wp.floor(p[0]), wp.floor(p[1]), wp.floor(p[2]))
    f = fract_vec3(p)
    
    fx = f[0]*f[0]*(3.0 - 2.0*f[0])
    fy = f[1]*f[1]*(3.0 - 2.0*f[1])
    fz = f[2]*f[2]*(3.0 - 2.0*f[2])
    
    c000 = hash31(i + wp.vec3(0.0, 0.0, 0.0))
    c100 = hash31(i + wp.vec3(1.0, 0.0, 0.0))
    c010 = hash31(i + wp.vec3(0.0, 1.0, 0.0))
    c110 = hash31(i + wp.vec3(1.0, 1.0, 0.0))
    c001 = hash31(i + wp.vec3(0.0, 0.0, 1.0))
    c101 = hash31(i + wp.vec3(1.0, 0.0, 1.0))
    c011 = hash31(i + wp.vec3(0.0, 1.0, 1.0))
    c111 = hash31(i + wp.vec3(1.0, 1.0, 1.0))
    
    c00 = wp.lerp(c000, c100, fx)
    c10 = wp.lerp(c010, c110, fx)
    c01 = wp.lerp(c001, c101, fx)
    c11 = wp.lerp(c011, c111, fx)
    
    c0 = wp.lerp(c00, c10, fy)
    c1 = wp.lerp(c01, c11, fy)
    return wp.lerp(c0, c1, fz)

@wp.func
def fbm_ridged_3d(p: wp.vec3):
    val = float(0.0)
    amp = float(0.5)
    freq = float(1.0)
    for _ in range(5):
        n = value_noise_3d(p * freq)
        val += amp * (1.0 - wp.abs(n - 0.5) * 2.0)
        amp *= 0.5
        freq *= 2.0
    return val

@wp.func
def blackbody(temp: float):
    # Simulates Planckian radiation color
    t = wp.clamp(temp, 1000.0, 40000.0) / 100.0
    
    r = float(1.0)
    if t > 66.0:
        r = wp.clamp(1.292936186 * wp.pow(t - 60.0, -0.1332047592), 0.0, 1.0)
        
    g = float(0.0)
    if t <= 66.0:
        g = wp.clamp(0.3900815787 * wp.log(t) - 0.6318414437, 0.0, 1.0)
    else:
        g = wp.clamp(1.12989086 * wp.pow(t - 60.0, -0.0755148492), 0.0, 1.0)
        
    b = float(1.0)
    if t <= 19.0:
        b = 0.0
    elif t < 66.0:
        b = wp.clamp(0.5432067891 * wp.log(t - 10.0) - 1.196254089, 0.0, 1.0)
        
    return wp.vec3(r, g, b)

# ---------------------------------------------------------
# GRID & FLUID SIMULATION KERNELS
# ---------------------------------------------------------

@wp.func
def grid_to_world_cylindrical(i: int, j: int, k: int, cfg: SimConfig):
    r = cfg.cylinder_min_radius + (float(i) + 0.5) * cfg.dr
    theta = (float(j) + 0.5) * cfg.dtheta - wp.PI
    y = -cfg.cylinder_half_height + (float(k) + 0.5) * cfg.dy
    return wp.vec3(r * wp.cos(theta), y, r * wp.sin(theta))

@wp.func
def world_to_grid_cylindrical(pos: wp.vec3, cfg: SimConfig):
    r = wp.length(wp.vec2(pos[0], pos[2]))
    theta = wp.atan2(pos[2], pos[0])
    y = pos[1]
    r_idx = (r - cfg.cylinder_min_radius) / cfg.dr - 0.5
    theta_idx = (theta + wp.PI) / cfg.dtheta - 0.5
    y_idx = (y + cfg.cylinder_half_height) / cfg.dy - 0.5
    return wp.vec3(r_idx, theta_idx, y_idx)

@wp.func
def sample_grid_f(field: wp.array3d[float], pos_grid: wp.vec3, grid_r: int, grid_theta: int, grid_y: int):
    x, y, z = pos_grid[0], pos_grid[1], pos_grid[2]
    x0 = int(wp.floor(x)); x1 = x0 + 1
    y0 = int(wp.floor(y)); y1 = y0 + 1
    z0 = int(wp.floor(z)); z1 = z0 + 1
    
    fx = x - float(x0); fy = y - float(y0); fz = z - float(z0)
    
    x0_c = wp.clamp(x0, 0, grid_r - 1); x1_c = wp.clamp(x1, 0, grid_r - 1)
    z0_c = wp.clamp(z0, 0, grid_y - 1); z1_c = wp.clamp(z1, 0, grid_y - 1)
    
    y0_c = y0 % grid_theta; 
    if y0_c < 0: y0_c += grid_theta
    y1_c = y1 % grid_theta; 
    if y1_c < 0: y1_c += grid_theta
    
    c00 = wp.lerp(field[x0_c, y0_c, z0_c], field[x1_c, y0_c, z0_c], fx)
    c10 = wp.lerp(field[x0_c, y1_c, z0_c], field[x1_c, y1_c, z0_c], fx)
    c01 = wp.lerp(field[x0_c, y0_c, z1_c], field[x1_c, y0_c, z1_c], fx)
    c11 = wp.lerp(field[x0_c, y1_c, z1_c], field[x1_c, y1_c, z1_c], fx)
    return wp.lerp(wp.lerp(c00, c10, fy), wp.lerp(c01, c11, fy), fz)

@wp.func
def sample_grid_v(field: wp.array3d[wp.vec3], pos_grid: wp.vec3, grid_r: int, grid_theta: int, grid_y: int):
    x, y, z = pos_grid[0], pos_grid[1], pos_grid[2]
    x0 = int(wp.floor(x)); x1 = x0 + 1
    y0 = int(wp.floor(y)); y1 = y0 + 1
    z0 = int(wp.floor(z)); z1 = z0 + 1
    
    fx = x - float(x0); fy = y - float(y0); fz = z - float(z0)
    
    x0_c = wp.clamp(x0, 0, grid_r - 1); x1_c = wp.clamp(x1, 0, grid_r - 1)
    z0_c = wp.clamp(z0, 0, grid_y - 1); z1_c = wp.clamp(z1, 0, grid_y - 1)
    
    y0_c = y0 % grid_theta; 
    if y0_c < 0: y0_c += grid_theta
    y1_c = y1 % grid_theta; 
    if y1_c < 0: y1_c += grid_theta
    
    c00 = wp.lerp(field[x0_c, y0_c, z0_c], field[x1_c, y0_c, z0_c], fx)
    c10 = wp.lerp(field[x0_c, y1_c, z0_c], field[x1_c, y1_c, z0_c], fx)
    c01 = wp.lerp(field[x0_c, y0_c, z1_c], field[x1_c, y0_c, z1_c], fx)
    c11 = wp.lerp(field[x0_c, y1_c, z1_c], field[x1_c, y1_c, z1_c], fx)
    return wp.lerp(wp.lerp(c00, c10, fy), wp.lerp(c01, c11, fy), fz)

@wp.kernel
def init_scene(density_field: wp.array3d[float], cfg: SimConfig):
    i, j, k = wp.tid()
    pos_world = grid_to_world_cylindrical(i, j, k, cfg)
    radius_xz = wp.length(wp.vec2(pos_world[0], pos_world[2]))
    
    vertical_noise_pos = pos_world * cfg.vertical_noise_scale
    vertical_mod = 1.0 + (fbm_ridged_3d(vertical_noise_pos) - 0.5) * 2.0 * cfg.vertical_strength
    modulated_half_thickness = cfg.cylinder_half_height * 0.15 * vertical_mod
    
    falloff_y = wp.smoothstep(modulated_half_thickness, modulated_half_thickness * 0.2, wp.abs(pos_world[1]))
    
    # INTERSTELLAR: Extremely sharp cut-off near the ISCO (Inner edge)
    falloff_in = wp.smoothstep(cfg.cylinder_min_radius, cfg.cylinder_min_radius * 1.1, radius_xz)
    falloff_out = wp.smoothstep(cfg.cylinder_max_radius, cfg.cylinder_max_radius * 0.5, radius_xz)
    base_density = falloff_y * falloff_in * falloff_out
    
    warp_coords = pos_world * cfg.warp_field_scale
    warp_vec = wp.vec3(
        fbm_ridged_3d(warp_coords + wp.vec3(13.7, 0.0, 0.0)),
        fbm_ridged_3d(warp_coords + wp.vec3(24.2, 0.0, 0.0)),
        fbm_ridged_3d(warp_coords + wp.vec3(19.1, 0.0, 0.0))
    )
    warped_pos = pos_world + (warp_vec - wp.vec3(0.5, 0.5, 0.5)) * 2.0 * cfg.warp_strength
    
    theta = wp.atan2(warped_pos[2], warped_pos[0])
    cos_t, sin_t = wp.cos(theta), wp.sin(theta)
    radial_dir = wp.vec3(cos_t, 0.0, sin_t)
    tangential_dir = wp.vec3(-sin_t, 0.0, cos_t)
    vertical_dir = wp.vec3(0.0, 1.0, 0.0)
    
    local_coords = wp.vec3(wp.dot(warped_pos, radial_dir), wp.dot(warped_pos, tangential_dir), wp.dot(warped_pos, vertical_dir))
    
    # INTERSTELLAR: Stretched orbital banding instead of puffy clouds
    ring_coords = wp.vec3(local_coords[0] * 5.0, local_coords[1] * 0.02, local_coords[2] * 4.0)
    filament_noise = wp.pow(fbm_ridged_3d(ring_coords), cfg.filament_contrast)
    
    clump_coords = local_coords * cfg.clump_noise_scale
    clump_noise = fbm_ridged_3d(clump_coords)
    combined_noise = wp.lerp(filament_noise, clump_noise, cfg.clump_strength)
    
    final_density = base_density * combined_noise * cfg.disk_noise_strength
    density_field[i, j, k] = wp.max(0.0, final_density)

@wp.kernel
def init_velocity(velocity_field: wp.array3d[wp.vec3], cfg: SimConfig):
    i, j, k = wp.tid()
    pos_world = grid_to_world_cylindrical(i, j, k, cfg)
    r = wp.length(wp.vec2(pos_world[0], pos_world[2]))
    ideal_vel = wp.vec3(0.0, 0.0, 0.0)
    if r > 0.1:
        tangential_dir = wp.normalize(wp.vec3(-pos_world[2], 0.0, pos_world[0]))
        speed = wp.sqrt(cfg.gm / (r + 0.1)) * cfg.orbital_velocity_scale
        ideal_vel = tangential_dir * speed
        
    theta = wp.atan2(pos_world[2], pos_world[0])
    cos_t, sin_t = wp.cos(theta), wp.sin(theta)
    v_r = ideal_vel[0] * cos_t + ideal_vel[2] * sin_t
    v_theta = -ideal_vel[0] * sin_t + ideal_vel[2] * cos_t
    velocity_field[i, j, k] = wp.vec3(v_r, v_theta, 0.0)

@wp.kernel
def bake_noise_kernel(noise_field: wp.array3d[float], cfg: SimConfig):
    # OPTIMIZATION: Pre-evaluates costly FBM structural carving mathematically 
    # across the grid mapping so we don’t calculate it per-pixel during ray marching.
    i, j, k = wp.tid()
    pos_world = grid_to_world_cylindrical(i, j, k, cfg)
    
    r = wp.length(wp.vec2(pos_world[0], pos_world[2]))
    theta = wp.atan2(pos_world[2], pos_world[0])
    twist = theta - r * 1.2 
    
    noise_pos_macro = wp.vec3(r * 2.5, pos_world[1] * 4.0, twist * 5.0)
    macro_noise = fbm_ridged_3d(noise_pos_macro)
    
    noise_pos_micro = wp.vec3(r * 8.0, pos_world[1] * 12.0, twist * 16.0)
    micro_noise = fbm_ridged_3d(noise_pos_micro)
    
    cloud_structure = wp.pow(macro_noise * 0.6 + micro_noise * 0.4, 1.8)
    noise_field[i, j, k] = cloud_structure

@wp.kernel
def advect_scalar(field_in: wp.array3d[float], field_out: wp.array3d[float], vel_field: wp.array3d[wp.vec3], dt: float, cfg: SimConfig):
    i, j, k = wp.tid()
    vel = vel_field[i, j, k]
    r = cfg.cylinder_min_radius + (float(i) + 0.5) * cfg.dr
    dr_grid = (vel[0] * dt) / cfg.dr
    d_theta_grid = ((vel[1] / (r + 1e-6)) * dt) / cfg.dtheta
    dy_grid = (vel[2] * dt) / cfg.dy
    
    p_prev = wp.vec3(float(i) - dr_grid, float(j) - d_theta_grid, float(k) - dy_grid)
    field_out[i, j, k] = sample_grid_f(field_in, p_prev, cfg.grid_r, cfg.grid_theta, cfg.grid_y)

@wp.kernel
def compute_bfecc_err_scalar(orig: wp.array3d[float], backward: wp.array3d[float], err: wp.array3d[float]):
    i, j, k = wp.tid()
    err[i, j, k] = (orig[i, j, k] - backward[i, j, k]) * 0.5
    
@wp.kernel
def apply_bfecc_err_scalar(orig: wp.array3d[float], err: wp.array3d[float], corr: wp.array3d[float]):
    i, j, k = wp.tid()
    corr[i, j, k] = orig[i, j, k] + err[i, j, k]

@wp.kernel
def advect_velocity(field_in: wp.array3d[wp.vec3], field_out: wp.array3d[wp.vec3], vel_field: wp.array3d[wp.vec3], dt: float, cfg: SimConfig):
    i, j, k = wp.tid()
    vel = vel_field[i, j, k]
    r = cfg.cylinder_min_radius + (float(i) + 0.5) * cfg.dr
    dr_grid = (vel[0] * dt) / cfg.dr
    d_theta_grid = ((vel[1] / (r + 1e-6)) * dt) / cfg.dtheta
    dy_grid = (vel[2] * dt) / cfg.dy
    
    p_prev = wp.vec3(float(i) - dr_grid, float(j) - d_theta_grid, float(k) - dy_grid)
    sampled = sample_grid_v(field_in, p_prev, cfg.grid_r, cfg.grid_theta, cfg.grid_y)
    field_out[i, j, k] = wp.lerp(sampled, wp.vec3(0.0, 0.0, 0.0), cfg.dissipation)

@wp.kernel
def apply_forces(vel: wp.array3d[wp.vec3], cfg: SimConfig):
    i, j, k = wp.tid()
    pos_world = grid_to_world_cylindrical(i, j, k, cfg)
    r = wp.length(wp.vec2(pos_world[0], pos_world[2]))
    
    ideal_speed = wp.sqrt(cfg.gm / (r + 0.1))
    tangential_dir = wp.normalize(wp.vec3(-pos_world[2], 0.0, pos_world[0]))
    ideal_vel = tangential_dir * (ideal_speed * cfg.orbital_velocity_scale)
    
    theta = wp.atan2(pos_world[2], pos_world[0])
    sin_t, cos_t = wp.sin(theta), wp.cos(theta)
    
    v = vel[i, j, k]
    cur_vel = wp.vec3(v[0]*cos_t - v[1]*sin_t, v[2], v[0]*sin_t + v[1]*cos_t)
    correction = ideal_vel - cur_vel
    
    grav_force = wp.vec3(0.0, 0.0, 0.0)
    r_sqr = wp.dot(pos_world, pos_world)
    if r_sqr > 0.1:
        grav_force = wp.normalize(pos_world) * (-cfg.gravity_strength * cfg.gm / r_sqr)
        
    total_force = correction * cfg.orbital_assist_strength + grav_force
    force_r = wp.dot(total_force, wp.vec3(cos_t, 0.0, sin_t))
    force_theta = wp.dot(total_force, wp.vec3(-sin_t, 0.0, cos_t))
    
    vel[i, j, k] = v + wp.vec3(force_r, force_theta, total_force[1]) * cfg.dt_sim

@wp.kernel
def clamp_velocity(vel: wp.array3d[wp.vec3], cfg: SimConfig):
    i, j, k = wp.tid()
    v = vel[i, j, k]
    if wp.dot(v, v) > cfg.max_velocity_physical * cfg.max_velocity_physical:
        vel[i, j, k] = wp.normalize(v) * cfg.max_velocity_physical

# ---------------------------------------------------------
# RAY MARCHING & RENDERING KERNELS
# ---------------------------------------------------------

@wp.func
def get_background_color_from_skybox(ray_dir: wp.vec3, skybox: wp.array2d[wp.vec3], w: int, h: int):
    phi = wp.acos(wp.clamp(ray_dir[1], -1.0, 1.0))
    theta = wp.atan2(ray_dir[2], ray_dir[0])
    u = (theta + wp.PI) / (2.0 * wp.PI)
    v = 1.0 - (phi / wp.PI)
    fw, fh = float(w), float(h)
    x = wp.clamp(u * fw - 0.5, 0.0, fw - 1.001)
    y = wp.clamp(v * fh - 0.5, 0.0, fh - 1.001)
    x0, y0 = int(x), int(y)
    x1, y1 = wp.min(x0 + 1, w - 1), wp.min(y0 + 1, h - 1)
    fx, fy = x - float(x0), y - float(y0)
    c0 = wp.lerp(skybox[y0, x0], skybox[y0, x1], fx)
    c1 = wp.lerp(skybox[y1, x0], skybox[y1, x1], fx)
    return wp.lerp(c0, c1, fy)

@wp.func
def get_acceleration_gr(pos: wp.vec3, vel: wp.vec3, rs: float):
    r = wp.length(pos) + 1e-9
    L_vec = wp.cross(pos, vel)
    gr_term = (3.0 * rs * wp.dot(L_vec, L_vec)) / (2.0 * wp.pow(r, 5.0))
    return pos * (-gr_term)

@wp.func
def dopri5_step(pos: wp.vec3, vel: wp.vec3, dt: float, rs: float):
    a21 = 1.0/5.0
    a31 = 3.0/40.0; a32 = 9.0/40.0
    a41 = 44.0/45.0; a42 = -56.0/15.0; a43 = 32.0/9.0
    a51 = 19372.0/6561.0; a52 = -25360.0/2187.0; a53 = 64448.0/6561.0; a54 = -212.0/729.0
    a61 = 9017.0/3168.0; a62 = -355.0/33.0; a63 = 46732.0/5247.0; a64 = 49.0/176.0; a65 = -5103.0/18656.0
    a71 = 35.0/384.0; a73 = 500.0/1113.0; a74 = 125.0/192.0; a75 = -2187.0/6784.0; a76 = 11.0/84.0
    b5_1 = 35.0/384.0; b5_3 = 500.0/1113.0; b5_4 = 125.0/192.0; b5_5 = -2187.0/6784.0; b5_6 = 11.0/84.0
    b4_1 = 5179.0/57600.0; b4_3 = 7571.0/16695.0; b4_4 = 393.0/640.0; b4_5 = -92097.0/339200.0; b4_6 = 187.0/2100.0; b4_7 = 1.0/40.0

    k1_pos = vel
    k1_vel = get_acceleration_gr(pos, vel, rs)
    k2_pos = vel + k1_vel * (dt * a21)
    k2_vel = get_acceleration_gr(pos + k1_pos * (dt * a21), k2_pos, rs)
    k3_pos = vel + k1_vel * (dt * a31) + k2_vel * (dt * a32)
    k3_vel = get_acceleration_gr(pos + k1_pos * (dt * a31) + k2_pos * (dt * a32), k3_pos, rs)
    k4_pos = vel + k1_vel * (dt * a41) + k2_vel * (dt * a42) + k3_vel * (dt * a43)
    k4_vel = get_acceleration_gr(pos + k1_pos * (dt * a41) + k2_pos * (dt * a42) + k3_pos * (dt * a43), k4_pos, rs)
    k5_pos = vel + k1_vel * (dt * a51) + k2_vel * (dt * a52) + k3_vel * (dt * a53) + k4_vel * (dt * a54)
    k5_vel = get_acceleration_gr(pos + k1_pos * (dt * a51) + k2_pos * (dt * a52) + k3_pos * (dt * a53) + k4_pos * (dt * a54), k5_pos, rs)
    k6_pos = vel + k1_vel * (dt * a61) + k2_vel * (dt * a62) + k3_vel * (dt * a63) + k4_vel * (dt * a64) + k5_vel * (dt * a65)
    k6_vel = get_acceleration_gr(pos + k1_pos * (dt * a61) + k2_pos * (dt * a62) + k3_pos * (dt * a63) + k4_pos * (dt * a64) + k5_pos * (dt * a65), k6_pos, rs)
    k7_pos = vel + k1_vel * (dt * a71) + k3_vel * (dt * a73) + k4_vel * (dt * a74) + k5_vel * (dt * a75) + k6_vel * (dt * a76)
    
    pos_5 = pos + k1_pos * (dt * b5_1) + k3_pos * (dt * b5_3) + k4_pos * (dt * b5_4) + k5_pos * (dt * b5_5) + k6_pos * (dt * b5_6)
    vel_5 = vel + k1_vel * (dt * b5_1) + k3_vel * (dt * b5_3) + k4_vel * (dt * b5_4) + k5_vel * (dt * b5_5) + k6_vel * (dt * b5_6)
    pos_4 = pos + k1_pos * (dt * b4_1) + k3_pos * (dt * b4_3) + k4_pos * (dt * b4_4) + k5_pos * (dt * b4_5) + k6_pos * (dt * b4_6) + k7_pos * (dt * b4_7)
    
    return pos_5, vel_5, wp.length(pos_5 - pos_4)

@wp.func
def sample_world_density(pos_world: wp.vec3, density_field: wp.array3d[float], noise_field: wp.array3d[float], s_cfg: SimConfig, r_cfg: RenderConfig):
    r = wp.length(wp.vec2(pos_world[0], pos_world[2]))
    y_abs = wp.abs(pos_world[1])
    density = float(0.0)
    
    # 1. EARLY EXIT: Expanded bounds to allow for wispy atmospheric layers
    if r < s_cfg.cylinder_min_radius * 0.8 or r > s_cfg.cylinder_max_radius * 1.2 or y_abs > s_cfg.cylinder_half_height * 2.0:
        return 0.0

    # 2. HYDROSTATIC FLARING
    flare_factor = wp.clamp((r - s_cfg.cylinder_min_radius) / (s_cfg.cylinder_max_radius - s_cfg.cylinder_min_radius), 0.0, 1.0)
    scale_height = s_cfg.cylinder_half_height * (0.15 + 0.85 * wp.pow(flare_factor, 1.5))

    # 3. GAUSSIAN VERTICAL PROFILE
    vertical_profile = wp.exp(-(y_abs * y_abs) / (scale_height * scale_height * 0.5 + 1e-6))

    # Smooth radial bounds so the edges don't look like a cut pipe
    falloff_in = wp.smoothstep(s_cfg.cylinder_min_radius * 0.85, s_cfg.cylinder_min_radius * 1.2, r)
    falloff_out = wp.smoothstep(s_cfg.cylinder_max_radius * 1.2, s_cfg.cylinder_max_radius * 0.7, r)
    bounds_mask = falloff_in * falloff_out

    # Only process ray-marching math if we are inside the visible gas
    if bounds_mask > 1e-4 and vertical_profile > 1e-3:
        
        # Get the macro-scale shape from the fluid grid
        grid_pos = world_to_grid_cylindrical(pos_world, s_cfg)
        base_density = sample_grid_f(density_field, grid_pos, s_cfg.grid_r, s_cfg.grid_theta, s_cfg.grid_y)
        
        # FASTER ARCHITECTURE: Look up high frequency mask noise straight from VRAM 
        # instead of evaluating 5-octave FBM multiple times per pixel per step
        cloud_structure = sample_grid_f(noise_field, grid_pos, s_cfg.grid_r, s_cfg.grid_theta, s_cfg.grid_y)
        
        # 6. ATMOSPHERIC STRATIFICATION
        height_ratio = wp.clamp(y_abs / scale_height, 0.0, 1.0)
        turbulence_mix = wp.lerp(0.2, 1.0, height_ratio) # 20% noise at the core, 100% noise at the edges
        
        structural_density = wp.lerp(1.0, cloud_structure * 2.5, turbulence_mix)
        
        # Final combine: Grid Macro x Depth Profile x Cached Noise x Bounds
        density = base_density * vertical_profile * structural_density * bounds_mask

    return density

@wp.func
def get_disk_emission(pos: wp.vec3, photon_dir: wp.vec3, r_cfg: RenderConfig, s_cfg: SimConfig):
    r_xz = wp.length(wp.vec2(pos[0], pos[2]))
    
    # INTERSTELLAR: Blazing hot white core (12000K), cool outer edge (2000K)
    r_ratio = wp.clamp(r_cfg.disk_inner / (r_xz + 1e-6), 0.0, 1.0)
    temp_factor = wp.pow(r_ratio, 1.5)
    temp_kelvin = wp.lerp(2000.0, 12000.0, temp_factor) 
    
    speed = wp.sqrt(r_cfg.gm / (r_xz + 0.1)) * s_cfg.orbital_velocity_scale
    speed = wp.clamp(speed, 0.0, 0.95)
    
    tangential_dir = wp.normalize(wp.vec3(-pos[2], 0.0, pos[0]))
    vel_world = tangential_dir * speed
    
    # INTERSTELLAR: Proper Relativistic Beaming using the *bent* photon direction
    beta_val = speed
    gamma = 1.0 / wp.sqrt(1.0 - beta_val*beta_val)
    beta_cos = wp.dot(vel_world, -photon_dir) 
    
    doppler = wp.clamp(1.0 / (gamma * (1.0 - beta_cos)), 0.1, 4.0)
    beaming = wp.pow(doppler, r_cfg.doppler_strength)
    
    shifted_temp = temp_kelvin * doppler
    base_color = blackbody(shifted_temp)
    
    shadow_falloff = wp.smoothstep(r_cfg.eq_shadow_width, 0.0, wp.abs(pos[1]))
    shadow_factor = 1.0 - r_cfg.eq_shadow_strength * shadow_falloff
    
    return base_color * beaming * shadow_factor * r_cfg.emission_strength

@wp.func
def march_volume_segment(start_pos: wp.vec3, end_pos: wp.vec3, transmittance_in: float, photon_dir: wp.vec3, density_field: wp.array3d[float], noise_field: wp.array3d[float], s_cfg: SimConfig, r_cfg: RenderConfig):
    color_out = wp.vec3(0.0, 0.0, 0.0)
    transmittance_out = transmittance_in
    segment_vec = end_pos - start_pos
    segment_len = wp.length(segment_vec)
    
    if segment_len > 1e-4:
        step_size = segment_len / float(r_cfg.volume_substeps)
        segment_ray_dir = segment_vec / segment_len
        
        for i in range(r_cfg.volume_substeps):
            if transmittance_out < 1e-3:
                break
            p = start_pos + segment_ray_dir * (float(i) + 0.5) * step_size
            density = sample_world_density(p, density_field, noise_field, s_cfg, r_cfg) * r_cfg.density_multiplier
            
            if density > 1e-3:
                density = wp.pow(density, r_cfg.density_pow)
                total_light = get_disk_emission(p, photon_dir, r_cfg, s_cfg)
                
                step_trans = wp.exp(-density * step_size * r_cfg.absorption_coefficient)
                color_out += total_light * (density * transmittance_out * step_size)
                transmittance_out *= step_trans
                
    return color_out, transmittance_out

@wp.kernel
def render_kernel(
    pixels_hdr: wp.array2d[wp.vec3],
    skybox: wp.array2d[wp.vec3],
    density_field: wp.array3d[float],
    noise_field: wp.array3d[float],
    skybox_w: int,
    skybox_h: int,
    cam_pos: wp.vec3,
    cam_to_world: wp.mat33,
    fov_local: float,
    width: int,
    height: int,
    frame_count: int,
    s_cfg: SimConfig,
    r_cfg: RenderConfig
):
    x, y = wp.tid()
    u = (float(x) - float(width) * 0.5) / float(height)
    v = (float(y) - float(height) * 0.5) / float(height)
    
    ray_dir = wp.normalize(cam_to_world * wp.normalize(wp.vec3(u, v, fov_local)))
    
    pos = cam_pos
    vel = ray_dir
    color = wp.vec3(0.0, 0.0, 0.0)
    transmittance = float(1.0)
    
    step = int(0)
    rng = wp.rand_init(frame_count, y * width + x)
    jitter = wp.randf(rng)
    dt = r_cfg.dt_initial * (0.8 + 0.4 * jitter)
    hit_object = int(0)
    
    while step < r_cfg.max_steps and hit_object == 0 and transmittance > 1e-3:
        r = wp.length(pos)
        if r <= r_cfg.horizon_radius + 0.001:
            hit_object = 1
        elif r > r_cfg.far_field:
            hit_object = 1
            color += get_background_color_from_skybox(vel, skybox, skybox_w, skybox_h) * transmittance
        else:
            pos_new, vel_new, error = dopri5_step(pos, vel, dt, r_cfg.rs)
            
            # Only accumulate volume if the physics step was successful
            if error <= r_cfg.tolerance:
                # Pass wp.normalize(vel) so Doppler calculates against the bent geodesic
                color_gas, trans_new = march_volume_segment(pos, pos_new, transmittance, wp.normalize(vel), density_field, noise_field, s_cfg, r_cfg)
                color += color_gas
                transmittance = trans_new
                
                pos = pos_new
                vel = vel_new
                step += 1
                
            error_clamped = wp.max(error, 1e-12)
            scale = 0.9 * wp.pow(r_cfg.tolerance / error_clamped, 1.0 / 6.0)
            dt = wp.clamp(dt * scale, r_cfg.dt_min, r_cfg.dt_max)
                
    if hit_object == 0:
        color += get_background_color_from_skybox(vel, skybox, skybox_w, skybox_h) * transmittance

    pixels_hdr[y, x] = color

# ---------------------------------------------------------
# CINEMATIC POST-PROCESSING KERNELS
# ---------------------------------------------------------

@wp.func
def tone_map_aces(color: wp.vec3):
    A = 2.51; B = 0.03; C = 2.43; D = 0.59; E = 0.14
    r = wp.clamp((color[0] * (A * color[0] + B)) / (color[0] * (C * color[0] + D) + E), 0.0, 1.0)
    g = wp.clamp((color[1] * (A * color[1] + B)) / (color[1] * (C * color[1] + D) + E), 0.0, 1.0)
    b = wp.clamp((color[2] * (A * color[2] + B)) / (color[2] * (C * color[2] + D) + E), 0.0, 1.0)
    return wp.vec3(r, g, b)

@wp.kernel
def extract_bright_kernel(hdr: wp.array2d[wp.vec3], bright: wp.array2d[wp.vec3]):
    x, y = wp.tid()
    c = hdr[y, x]
    brightness = wp.dot(c, wp.vec3(0.299, 0.587, 0.114))
    mask = wp.smoothstep(1.0, 2.5, brightness)
    bright[y, x] = c * mask

@wp.kernel
def blur_h_kernel(img_in: wp.array2d[wp.vec3], img_out: wp.array2d[wp.vec3], width: int, radius: int):
    x, y = wp.tid()
    c = wp.vec3(0.0, 0.0, 0.0)
    weight_sum = float(0.0)
    for i in range(-radius, radius + 1):
        xi = wp.clamp(x + i, 0, width - 1)
        w = wp.exp(-float(i*i) / float(radius*radius) * 2.0)
        c += img_in[y, xi] * w
        weight_sum += w
    img_out[y, x] = c / weight_sum

@wp.kernel
def blur_v_kernel(img_in: wp.array2d[wp.vec3], img_out: wp.array2d[wp.vec3], height: int, radius: int):
    x, y = wp.tid()
    c = wp.vec3(0.0, 0.0, 0.0)
    weight_sum = float(0.0)
    for i in range(-radius, radius + 1):
        yi = wp.clamp(y + i, 0, height - 1)
        w = wp.exp(-float(i*i) / float(radius*radius) * 2.0)
        c += img_in[yi, x] * w
        weight_sum += w
    img_out[y, x] = c / weight_sum

@wp.kernel
def composite_kernel(
    hdr: wp.array2d[wp.vec3],
    bloom: wp.array2d[wp.vec3],
    pixels_out: wp.array2d[wp.vec3],
    width: int,
    height: int
):
    x, y = wp.tid()
    u = (float(x) - float(width) * 0.5) / float(height)
    v = (float(y) - float(height) * 0.5) / float(height)
    
    offset = wp.length(wp.vec2(u, v)) * 0.0015
    dx = int(offset * float(width))
    
    x_r = wp.clamp(x - dx, 0, width - 1)
    x_b = wp.clamp(x + dx, 0, width - 1)
    
    c_r = hdr[y, x_r][0] + bloom[y, x_r][0] * 0.4
    c_g = hdr[y, x][1]   + bloom[y, x][1]   * 0.4
    c_b = hdr[y, x_b][2] + bloom[y, x_b][2] * 0.4
    c = wp.vec3(c_r, c_g, c_b)
    
    vignette = 1.0 - (u*u + v*v * (1.777 * 1.777)) * 0.3
    c = c * wp.max(vignette, 0.0)
    
    final = tone_map_aces(c * 0.95)
    pixels_out[y, x] = wp.vec3(wp.pow(final[0], 1.0/2.2), wp.pow(final[1], 1.0/2.2), wp.pow(final[2], 1.0/2.2))

# ---------------------------------------------------------
# RENDERER CLASS
# ---------------------------------------------------------

class SagittariusRenderer:
    def __init__(self, width=1280, show_gui=True, save_video_path=None, video_fps=24, use_caching=False):
        self.show_gui = show_gui
        self.save_video = save_video_path is not None
        self.use_caching = use_caching
        self.frame_count = 0
        
        self.WIDTH = width
        self.ASPECT_RATIO = 16.0 / 9.0
        self.HEIGHT = int(self.WIDTH / self.ASPECT_RATIO)
        
        self.cfg_render = RenderConfig()
        self.cfg_render.gm = 0.5
        self.cfg_render.rs = 2.0 * self.cfg_render.gm
        self.cfg_render.horizon_radius = self.cfg_render.rs
        
        # INTERSTELLAR: Disk sits safely outside ISCO (3x Rs) for the distinct shadow gap
        self.cfg_render.disk_inner = self.cfg_render.rs * 3.0
        self.cfg_render.disk_outer = self.cfg_render.rs * 15.0
        self.cfg_render.disk_half_thickness = (self.cfg_render.disk_outer - self.cfg_render.disk_inner) * 0.02
        
        self.cfg_render.far_field = 80.0
        self.cfg_render.volume_substeps = 64
        self.cfg_render.density_multiplier = 0.6
        self.cfg_render.density_pow = 1.5
        self.cfg_render.emission_strength = 25.0
        self.cfg_render.absorption_coefficient = 8.0
        
        self.cfg_render.doppler_strength = 2
        self.cfg_render.eq_shadow_width = 0.08
        self.cfg_render.eq_shadow_strength = 0.2 # Softened equatorial band
        
        self.cfg_render.dt_initial = 0.4
        self.cfg_render.dt_min = 1.0e-4
        self.cfg_render.dt_max = 1.0
        self.cfg_render.max_steps = 800
        self.cfg_render.tolerance = 1.0e-5

        self.cfg_sim = SimConfig()
        self.cfg_sim.cylinder_min_radius = self.cfg_render.disk_inner
        self.cfg_sim.cylinder_max_radius = self.cfg_render.disk_outer
        self.cfg_sim.cylinder_half_height = self.cfg_render.disk_half_thickness * 5.0
        
        self.cfg_sim.grid_r = 512
        self.cfg_sim.grid_theta = 1024
        self.cfg_sim.grid_y = 64
        self.cfg_sim.dr = (self.cfg_sim.cylinder_max_radius - self.cfg_sim.cylinder_min_radius) / self.cfg_sim.grid_r
        self.cfg_sim.dtheta = (2 * math.pi) / self.cfg_sim.grid_theta
        self.cfg_sim.dy = (2 * self.cfg_sim.cylinder_half_height) / self.cfg_sim.grid_y
        self.cfg_sim.dt_sim = 0.005
        self.cfg_sim.advection_strength = 1e-5
        self.cfg_sim.dissipation = 0.01 
        self.cfg_sim.orbital_velocity_scale = 1.0
        self.cfg_sim.gravity_strength = 1.0
        self.cfg_sim.orbital_assist_strength = 0.03
        self.cfg_sim.max_velocity_physical = 2.0 * self.cfg_sim.dr / self.cfg_sim.dt_sim
        self.cfg_sim.gm = self.cfg_render.gm
        
        # INTERSTELLAR: Turbulence tuned for stretched, cinematic orbital bands
        self.cfg_sim.vertical_noise_scale = 0.1
        self.cfg_sim.vertical_strength = 0.9
        self.cfg_sim.warp_field_scale = 1.0
        self.cfg_sim.warp_strength = 1.8
        self.cfg_sim.tangential_stretch = 90.0
        self.cfg_sim.filament_noise_scale = 1.5
        self.cfg_sim.filament_contrast = 6.0
        self.cfg_sim.clump_noise_scale = 0.1
        self.cfg_sim.clump_strength = 0.2
        self.cfg_sim.disk_noise_strength = 1.5

        self.pixels_device = wp.zeros(shape=(self.HEIGHT, self.WIDTH), dtype=wp.vec3, device="cuda")
        self.pixels_hdr = wp.zeros(shape=(self.HEIGHT, self.WIDTH), dtype=wp.vec3, device="cuda")
        self.bright_pass = wp.zeros(shape=(self.HEIGHT, self.WIDTH), dtype=wp.vec3, device="cuda")
        self.blur_h = wp.zeros(shape=(self.HEIGHT, self.WIDTH), dtype=wp.vec3, device="cuda")
        self.blur_v = wp.zeros(shape=(self.HEIGHT, self.WIDTH), dtype=wp.vec3, device="cuda")

        sim_dim = (self.cfg_sim.grid_r, self.cfg_sim.grid_theta, self.cfg_sim.grid_y)
        self.density_field = wp.zeros(sim_dim, dtype=float, device="cuda")
        self.density_1 = wp.zeros(sim_dim, dtype=float, device="cuda")
        self.density_2 = wp.zeros(sim_dim, dtype=float, device="cuda")
        self.density_err = wp.zeros(sim_dim, dtype=float, device="cuda")
        self.density_corr = wp.zeros(sim_dim, dtype=float, device="cuda")
        self.new_density_field = wp.zeros(sim_dim, dtype=float, device="cuda")
        
        self.noise_field = wp.zeros(sim_dim, dtype=float, device="cuda")
        
        self.velocity_field = wp.zeros(sim_dim, dtype=wp.vec3, device="cuda")
        self.new_velocity_field = wp.zeros(sim_dim, dtype=wp.vec3, device="cuda")
        
        self._ensure_skybox("starmap.jpg")
        self._load_skybox("starmap.jpg")
        self._init_fluid_grids()
        
        self.final_video_path = save_video_path
        self.video_fps = video_fps
        self.video_writer = None
        
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.final_video_path, fourcc, self.video_fps, (self.WIDTH, self.HEIGHT))
            
        self.cache_dir = "frame_cache"
        self.config_hash = self._get_config_hash()
        self.active_cache_dir = os.path.join(self.cache_dir, self.config_hash)
        if self.use_caching:
            os.makedirs(self.active_cache_dir, exist_ok=True)
            
        self._print_init_summary()

    def _ensure_skybox(self, path):
        if not os.path.exists(path):
            img = np.zeros((1024, 2048, 3), dtype=np.uint8)
            for _ in range(1500):
                x, y = np.random.randint(0, 2048), np.random.randint(0, 1024)
                c = np.random.randint(100, 255)
                img[y:min(y+2,1024), x:min(x+2,2048)] = (c, c, c)
            cv2.imwrite(path, img)

    def _load_skybox(self, path):
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_np = (img_rgb.astype(np.float32) / 255.0)
        self.skybox_w, self.skybox_h = img_np.shape[1], img_np.shape[0]
        self.skybox_device = wp.array(img_np.reshape((self.skybox_h, self.skybox_w, 3)), dtype=wp.vec3, device="cuda")

    def _init_fluid_grids(self):
        dim = (self.cfg_sim.grid_r, self.cfg_sim.grid_theta, self.cfg_sim.grid_y)
        wp.launch(init_scene, dim=dim, inputs=[self.density_field, self.cfg_sim], device="cuda")
        wp.launch(init_velocity, dim=dim, inputs=[self.velocity_field, self.cfg_sim], device="cuda")
        wp.launch(bake_noise_kernel, dim=dim, inputs=[self.noise_field, self.cfg_sim], device="cuda")
        wp.synchronize_device("cuda")

    def simulation_step(self):
        dim = (self.cfg_sim.grid_r, self.cfg_sim.grid_theta, self.cfg_sim.grid_y)
        
        wp.launch(advect_scalar, dim=dim, inputs=[self.density_field, self.density_1, self.velocity_field, self.cfg_sim.dt_sim, self.cfg_sim], device="cuda")
        wp.launch(advect_scalar, dim=dim, inputs=[self.density_1, self.density_2, self.velocity_field, -self.cfg_sim.dt_sim, self.cfg_sim], device="cuda")
        wp.launch(compute_bfecc_err_scalar, dim=dim, inputs=[self.density_field, self.density_2, self.density_err], device="cuda")
        wp.launch(apply_bfecc_err_scalar, dim=dim, inputs=[self.density_field, self.density_err, self.density_corr], device="cuda")
        wp.launch(advect_scalar, dim=dim, inputs=[self.density_corr, self.new_density_field, self.velocity_field, self.cfg_sim.dt_sim, self.cfg_sim], device="cuda")

        wp.launch(advect_velocity, dim=dim, inputs=[self.velocity_field, self.new_velocity_field, self.velocity_field, self.cfg_sim.dt_sim, self.cfg_sim], device="cuda")
                
        self.velocity_field, self.new_velocity_field = self.new_velocity_field, self.velocity_field
        self.density_field, self.new_density_field = self.new_density_field, self.density_field
        
        wp.launch(apply_forces, dim=dim, inputs=[self.velocity_field, self.cfg_sim], device="cuda")
        wp.launch(clamp_velocity, dim=dim, inputs=[self.velocity_field, self.cfg_sim], device="cuda")

    def _get_config_hash(self):
        params = {'WIDTH': self.WIDTH, 'HEIGHT': self.HEIGHT, 'GM': self.cfg_render.gm}
        return hashlib.sha256(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()

    def _get_frame_hash(self, cam_pos, cam_fwd, cam_up, fov):
        return hashlib.sha256(f"{self.frame_count}-{list(cam_pos)}-{list(cam_fwd)}-{list(cam_up)}-{fov:.4f}".encode()).hexdigest()

    def _print_init_summary(self):
        if not _rich_available:
            print(f"--- Cinematic Sagittarius Initialized ---\nResolution: {self.WIDTH}x{self.HEIGHT}")
            return
        settings_text = Text.from_markup(f"""
[bold]Resolution:[/bold]   {self.WIDTH}x{self.HEIGHT}
[bold]Physics:[/bold]      BFECC Advection, Clamped Doppler Beaming, Blackbody Radiation
[bold]Video Output:[/bold] {f'[cyan]"{self.final_video_path}"[/cyan]' if self.save_video else '[dim]Disabled[/dim]'}
[bold]Caching:[/bold]      {f'[green]ENABLED[/green]' if self.use_caching else '[yellow]DISABLED[/yellow]'}
""")
        console.print(Panel(settings_text, title="[bold blue]Cinematic Sagittarius Initialized[/bold blue]", border_style="blue"))

    def get_init_panel(self):
        if not _rich_available: return ""
        settings_text = Text.from_markup(f"""
[bold]Resolution:[/bold]   {self.WIDTH}x{self.HEIGHT}
[bold]Physics:[/bold]      BFECC Advection, Clamped Doppler Beaming, Blackbody Radiation
""")
        return Panel(settings_text, title="[bold blue]Cinematic Sagittarius[/bold blue]", border_style="blue", expand=False)

    def get_system_stats_str(self):
        if not _rich_available or not _psutil_available: return ""
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        return f"💻 CPU: {cpu: >4.1f}% | 🧠 RAM: {ram.percent: >4.1f}%"

    def step(self, cam_pos_np, cam_fwd_np, cam_up_np, fov):
        self.frame_count += 1
        self.simulation_step()
        
        cache_path = os.path.join(self.active_cache_dir, f"{self._get_frame_hash(cam_pos_np, cam_fwd_np, cam_up_np, fov)}.jpg")
        
        if self.use_caching and os.path.exists(cache_path):
            img_bgr = cv2.imread(cache_path)
            if self.save_video: self.video_writer.write(img_bgr)
            if self.show_gui: cv2.imshow("Sagittarius Black Hole", img_bgr); cv2.waitKey(1)
            return "cache_hit"

        fwd = cam_fwd_np / np.linalg.norm(cam_fwd_np)
        right = np.cross(fwd, cam_up_np)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)

        cam_pos = wp.vec3(cam_pos_np[0], cam_pos_np[1], cam_pos_np[2])
        cam_to_world = wp.mat33(
            right[0], up[0], fwd[0],
            right[1], up[1], fwd[1],
            right[2], up[2], fwd[2]
        )

        dim2d = (self.WIDTH, self.HEIGHT)
        
        wp.launch(
            kernel=render_kernel, dim=dim2d,
            inputs=[
                self.pixels_hdr, self.skybox_device, self.density_field,
                self.noise_field,
                self.skybox_w, self.skybox_h, cam_pos, cam_to_world, fov,
                self.WIDTH, self.HEIGHT, self.frame_count,
                self.cfg_sim, self.cfg_render
            ], device="cuda"
        )
        wp.launch(extract_bright_kernel, dim=dim2d, inputs=[self.pixels_hdr, self.bright_pass], device="cuda")
        wp.launch(blur_h_kernel, dim=dim2d, inputs=[self.bright_pass, self.blur_h, self.WIDTH, 24], device="cuda")
        wp.launch(blur_v_kernel, dim=dim2d, inputs=[self.blur_h, self.blur_v, self.HEIGHT, 24], device="cuda")
        wp.launch(composite_kernel, dim=dim2d, inputs=[self.pixels_hdr, self.blur_v, self.pixels_device, self.WIDTH, self.HEIGHT], device="cuda")
        
        wp.synchronize_device("cuda")

        img_255_rgb = np.clip(self.pixels_device.numpy() * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_255_rgb, cv2.COLOR_RGB2BGR)

        if self.use_caching: cv2.imwrite(cache_path, img_bgr)
        if self.save_video: self.video_writer.write(img_bgr)
        if self.show_gui: cv2.imshow("Sagittarius Black Hole", img_bgr); cv2.waitKey(1)

        return "cache_miss"

    def close(self):
        if self.video_writer:
            self.video_writer.release()
            if _rich_available:
                console.print(f"\n[bold green]✓ Video saved to '{self.final_video_path}'![/bold green]")
            else:
                print(f"\nVideo saved to {self.final_video_path}")
        if self.show_gui:
            cv2.destroyAllWindows()


# ---------------------------------------------------------
# EXECUTION SCRIPT
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Initializing Cinematic Gargantua Renderer...")
    # Instantiate renderer at a good preview resolution. Disabling live GUI since we just want to save the final frame
    renderer = SagittariusRenderer(width=1920, show_gui=False, use_caching=False)
    
    # Standard cinematic framing. Slightly above the disc, looking down/forward at the black hole core
    cam_pos = np.array([0.0, 2.5, -28.0])
    cam_fwd = np.array([0.0, -0.05, 1.0])
    cam_up  = np.array([0.0, 1.0, 0.0])
    fov_val = 1.0  # Approx 60 degrees FOV
    
    print("Rendering frame (this may take a few seconds depending on GPU)...")
    renderer.step(cam_pos, cam_fwd, cam_up, fov_val)
    
    # Fetch result from the GPU
    img_rgb = np.clip(renderer.pixels_device.numpy() * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    out_filename = "gargantua_render.jpg"
    cv2.imwrite(out_filename, img_bgr)
    print(f"\nSuccess! Final image saved to: {out_filename}")