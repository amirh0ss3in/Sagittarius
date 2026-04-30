# --- START OF FILE camera_controller.py ---

import time
import math
import numpy as np
from sagittarius_renderer import SagittariusRenderer

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
    from rich.live import Live
    _rich_available = True
    console = Console()
except ImportError:
    _rich_available = False
    class _DummyConsole:
        def print(self, text): print(text)
    console = _DummyConsole()

def get_camera_vectors_at_time(animation_time):
    # INTERSTELLAR: Orbit slowed down for cinematic gravity, camera pushed in
    orbit_period = 200.0
    phi = (animation_time / orbit_period) * 2.0 * math.pi + math.pi

    start_radius, end_radius = 45.0, 20.0
    animation_duration = 60.0
    progress = min(animation_time / animation_duration, 1.0)
    
    eased_progress = 0.5 * (1.0 - math.cos(progress * math.pi))
    radius = start_radius + (end_radius - start_radius) * eased_progress
    
    # INTERSTELLAR: Camera angle dropped to ~5.7 degrees above the accretion disk
    # This grazing angle forces the light to bend massively to reach the camera, 
    # creating the iconic "Halo" over the top and bottom of the event horizon.
    start_theta = math.pi / 2.0 - 0.1 
    end_theta = math.pi / 2.0 - 0.05
    theta = start_theta + (end_theta - start_theta) * eased_progress

    start_fov, end_fov = 1.0, 1.6
    fov = start_fov + (end_fov - start_fov) * eased_progress

    cam_x = radius * math.sin(theta) * math.cos(phi)
    cam_y = radius * math.cos(theta)
    cam_z = radius * math.sin(theta) * math.sin(phi)
    
    cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
    cam_fwd = -cam_pos 
    cam_fwd = cam_fwd / np.linalg.norm(cam_fwd)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    return cam_pos, cam_fwd, world_up, fov

def main():
    # RENDER CONFIGURATION
    SHOW_GUI = False       
    SAVE_VIDEO = True     

    VIDEO_DURATION_SECONDS = 40   
    VIDEO_FPS = 30
    OUTPUT_FILENAME = "sagittarius_flight.mp4"

    RENDER_WIDTH = 1920

    # Initialize the scene renderer
    scene = SagittariusRenderer(
        show_gui=SHOW_GUI,
        save_video_path=OUTPUT_FILENAME if SAVE_VIDEO else None,
        video_fps=VIDEO_FPS,
        width=RENDER_WIDTH,
        use_caching=True
    )
    
    total_frames = int(VIDEO_DURATION_SECONDS * VIDEO_FPS)
    cache_hits, cache_misses = 0, 0
    total_render_time = 0.0

    # Main application loop
    if _rich_available:
        init_panel = scene.get_init_panel() 
        job_text = Text.from_markup(f"[bold]Total Frames:[/bold] {total_frames}\n[bold]Output File:[/bold]  [cyan]'{OUTPUT_FILENAME}'[/cyan]")
        job_panel = Panel(job_text, title="[bold magenta]Render Job Started[/bold magenta]", border_style="magenta", expand=False)

        progress_columns = [
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ]
        progress = Progress(*progress_columns, transient=True) 

        render_group = Group(init_panel, job_panel, progress)

        with Live(render_group, console=console, refresh_per_second=10) as live:
            task = progress.add_task("Starting...", total=total_frames)

            for frame in range(total_frames):
                start_frame_time = time.time()
                
                animation_time = frame / VIDEO_FPS
                cam_pos, cam_fwd, world_up, fov = get_camera_vectors_at_time(animation_time)
                
                status = scene.step(cam_pos, cam_fwd, world_up, fov)
                
                frame_duration = time.time() - start_frame_time
                total_render_time += frame_duration
                
                if status == 'cache_hit': cache_hits += 1
                else: cache_misses += 1

                stats_str = scene.get_system_stats_str()
                cache_str = f"Cache: [green]✔ {cache_hits}[/green] [red]✖ {cache_misses}[/red]"

                description_text = Text.from_markup(
                    f"[cyan]Rendering Frame {frame + 1}/{total_frames}[/cyan]\n"
                    f"[dim]  └─ Last: {frame_duration:.2f}s | {cache_str} | {stats_str}[/dim]"
                )
                
                progress.update(task, advance=1, description=description_text)
    else:
        print(f"\nStarting render of {total_frames} frames to '{OUTPUT_FILENAME}'...")
        for frame in range(total_frames):
            print(f"Rendering frame {frame+1}/{total_frames}...")
            animation_time = frame / VIDEO_FPS
            cam_pos, cam_fwd, world_up, fov = get_camera_vectors_at_time(animation_time)
            status = scene.step(cam_pos, cam_fwd, world_up, fov)

    # Clean up and finalize resources
    scene.close()

    # Print final summary
    if _rich_available:
        summary_table = Table(show_header=False, box=None, padding=(0, 2))
        summary_table.add_column(style="bold blue")
        summary_table.add_column()
        summary_table.add_row("Total Frames:", f"{total_frames}")
        summary_table.add_row("Total Time:", f"{total_render_time:.2f} seconds")
        summary_table.add_row("Average Time:", f"{(total_render_time/total_frames):.2f} s/frame" if total_frames > 0 else "N/A")
        
        console.print(Panel(summary_table, title="[bold blue]Render Summary[/bold blue]", border_style="blue", expand=False))

if __name__ == "__main__":
    main()