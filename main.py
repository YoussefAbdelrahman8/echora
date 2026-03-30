# =============================================================================
# main.py — ECHORA Entry Point
# =============================================================================
# The single file you run to start ECHORA.
#
#   python main.py
#
# Responsibilities:
#   1. Parse command-line arguments (debug mode, no-display, etc.)
#   2. Set up logging to both console and file
#   3. Print startup banner
#   4. Create and run the ControlUnit
#   5. Handle all top-level errors cleanly
#   6. Write a final session report on exit
# =============================================================================


# =============================================================================
# WINDOWS PROCESS PRIORITY — must be absolute first line of execution
# =============================================================================
import sys
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x00000080   # HIGH_PRIORITY_CLASS
    )


# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """
    Parses command-line arguments.

    Usage examples:
        python main.py                    # normal run
        python main.py --debug            # verbose logging
        python main.py --no-display       # no debug window (wearable mode)
        python main.py --no-audio         # mute all speech (testing)
        python main.py --log-file         # save log to file
    """

    parser = argparse.ArgumentParser(
        description="ECHORA — AI-Powered Sensory Substitution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  Normal run with debug window
  python main.py --no-display     Wearable mode — no screen needed
  python main.py --debug          Verbose logging for development
  python main.py --log-file       Also save logs to logs/session.log
        """
    )

    parser.add_argument(
        "--debug",
        action  = "store_true",
        help    = "Enable verbose DEBUG logging (default: INFO only)"
    )

    parser.add_argument(
        "--no-display",
        action  = "store_true",
        dest    = "no_display",
        help    = "Disable debug window — runs headlessly (wearable mode)"
    )

    parser.add_argument(
        "--no-audio",
        action  = "store_true",
        dest    = "no_audio",
        help    = "Disable all audio output (useful for testing)"
    )

    parser.add_argument(
        "--log-file",
        action  = "store_true",
        dest    = "log_file",
        help    = "Save log output to logs/session_TIMESTAMP.log"
    )

    parser.add_argument(
        "--tolerance",
        type    = float,
        default = None,
        help    = "Face recognition tolerance 0.0-1.0 (default: from config)"
    )

    return parser.parse_args()


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(debug: bool = False, log_file: bool = False):
    """
    Configures the root logger for the entire ECHORA system.

    All modules use logger = logging.getLogger("ECHORA") from utils.py.
    This function configures that logger centrally.

    Arguments:
        debug:    if True, set level to DEBUG (very verbose)
                  if False, set level to INFO (normal operation)
        log_file: if True, also write logs to a timestamped file
    """

    level = logging.DEBUG if debug else logging.INFO

    # Get the ECHORA logger — same one used by all modules.
    echora_logger = logging.getLogger("ECHORA")
    echora_logger.setLevel(level)

    # Remove any existing handlers — prevents duplicate log lines
    # if this function is called more than once.
    echora_logger.handlers.clear()

    # ── Console handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Force UTF-8 on Windows to handle special characters in log messages.
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

    console_formatter = logging.Formatter(
        fmt     = "%(asctime)s — %(levelname)s — %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    echora_logger.addHandler(console_handler)

    # ── File handler (optional) ────────────────────────────────────────────────
    if log_file:
        # Create logs directory if it doesn't exist.
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Timestamped filename — one log file per session.
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path   = logs_dir / f"session_{timestamp}.log"

        file_handler = logging.FileHandler(str(log_path), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)   # always DEBUG in file

        file_formatter = logging.Formatter(
            fmt     = "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        echora_logger.addHandler(file_handler)

        print(f"  Log file: {log_path}")

    return echora_logger


# =============================================================================
# STARTUP BANNER
# =============================================================================

def print_banner():
    """Prints the ECHORA startup banner."""

    print()
    print("=" * 62)
    print("  ███████╗ ██████╗██╗  ██╗ ██████╗ ██████╗  █████╗  ")
    print("  ██╔════╝██╔════╝██║  ██║██╔═══██╗██╔══██╗██╔══██╗ ")
    print("  █████╗  ██║     ███████║██║   ██║██████╔╝███████║ ")
    print("  ██╔══╝  ██║     ██╔══██║██║   ██║██╔══██╗██╔══██║ ")
    print("  ███████╗╚██████╗██║  ██║╚██████╔╝██║  ██║██║  ██║ ")
    print("  ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝")
    print()
    print("  AI-Powered Sensory Substitution System")
    print("  For Blind and Visually Impaired Users")
    print("=" * 62)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()


# =============================================================================
# SESSION REPORT
# =============================================================================

def write_session_report(
    cu,
    start_time: float,
    exit_reason: str
):
    """
    Writes a brief session report to the logs folder.

    Records: runtime, frames processed, FPS, mode switches, errors.
    Useful for reviewing system performance after a session.

    Arguments:
        cu:          the ControlUnit instance (may be None if startup failed)
        start_time:  time.time() at session start
        exit_reason: why the session ended
    """

    try:
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = logs_dir / f"report_{timestamp}.txt"

        duration = time.time() - start_time
        hours    = int(duration // 3600)
        minutes  = int((duration % 3600) // 60)
        seconds  = int(duration % 60)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("ECHORA Session Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Date:        {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"Start time:  {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}\n")
            f.write(f"Duration:    {hours:02d}:{minutes:02d}:{seconds:02d}\n")
            f.write(f"Exit reason: {exit_reason}\n\n")

            if cu is not None and cu._started:
                # Performance stats from control unit.
                frame_count = cu._frame_count
                slow_frames = cu._slow_frames

                if cu._frame_times:
                    avg_ms = sum(cu._frame_times) / len(cu._frame_times)
                    fps    = 1000.0 / max(avg_ms, 1)
                else:
                    avg_ms = 0
                    fps    = 0

                f.write("Performance\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total frames:  {frame_count}\n")
                f.write(f"Average FPS:   {fps:.1f}\n")
                f.write(f"Avg frame ms:  {avg_ms:.1f}\n")
                f.write(f"Slow frames:   {slow_frames}\n\n")

                # State machine stats.
                if cu._state_machine:
                    sm_stats = cu._state_machine.get_stats()
                    f.write("State Machine\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Mode switches: {sm_stats.get('total_switches', 0)}\n")
                    f.write(f"Final mode:    {sm_stats.get('current_mode', 'N/A')}\n\n")

                # Database stats.
                try:
                    from database import get_db
                    db = get_db()
                    if db:
                        db_stats = db.get_stats()
                        f.write("Database\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"Registered persons: {db_stats.get('persons', 0)}\n")
                        f.write(f"Event log entries:  {db_stats.get('events', 0)}\n\n")
                except Exception:
                    pass

            else:
                f.write("System did not start successfully.\n")

        print(f"\n  Session report saved: {report_path}")

    except Exception as e:
        print(f"\n  Could not write session report: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    The main entry point for ECHORA.

    Parses arguments, sets up logging, creates the ControlUnit,
    runs the main loop, and handles shutdown cleanly.
    """

    # ── Parse arguments ────────────────────────────────────────────────────────
    args = parse_args()

    # ── Print banner ───────────────────────────────────────────────────────────
    print_banner()

    # ── Setup logging ──────────────────────────────────────────────────────────
    logger = setup_logging(
        debug    = args.debug,
        log_file = args.log_file
    )

    # ── Apply argument overrides to config ─────────────────────────────────────
    # These override config.py values based on command-line flags.
    # We do this BEFORE importing control_unit so the settings propagate.

    if args.no_display:
        # Monkey-patch the debug window flag in control_unit module.
        # This avoids needing to pass arguments through every constructor.
        import control_unit as cu_module
        cu_module.SHOW_DEBUG_WINDOW = False
        print("  Display: OFF (wearable mode)")
    else:
        print("  Display: ON  (press Q to quit)")

    if args.no_audio:
        logger.warning(
            "Audio disabled via --no-audio flag. "
            "TTS and spatial alerts will not play."
        )

    if args.tolerance is not None:
        import config
        config.FACE_RECOGNITION_TOLERANCE = args.tolerance
        print(f"  Face tolerance: {args.tolerance}")

    if args.debug:
        print("  Logging: DEBUG (verbose)")
    else:
        print("  Logging: INFO")

    print()

    # ── Record start time ──────────────────────────────────────────────────────
    start_time   = time.time()
    exit_reason  = "unknown"
    cu           = None

    # ── Import and run ControlUnit ─────────────────────────────────────────────
    try:
        from control_unit import ControlUnit

        cu = ControlUnit()

        # Override audio if --no-audio flag was passed.
        if args.no_audio:
            # Patch audio after ControlUnit is created but before startup.
            # We replace the audio init with a no-op version.
            class _SilentAudio:
                _ready = True
                def speak(self, *a, **kw): pass
                def announce_obstacle(self, *a, **kw): pass
                def announce_scene(self, *a, **kw): pass
                def announce_ocr(self, *a, **kw): pass
                def announce_face(self, *a, **kw): pass
                def announce_banknote(self, *a, **kw): pass
                def announce_mode_change(self, *a, **kw): pass
                def play_spatial_alert(self, *a, **kw): pass
                def stop_all(self, *a, **kw): pass
                def release(self): pass

            # We monkey-patch after startup by replacing _audio post-init.
            # Store the flag for after startup.
            _silent_audio = True
        else:
            _silent_audio = False

        # Start all sub-systems.
        cu.startup()

        # Apply silent audio after startup if requested.
        if _silent_audio:
            cu._audio = _SilentAudio()

        # Run the main loop — blocks until Q pressed or error.
        cu.run()

        exit_reason = "user_quit"

    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        print("\n\nInterrupted by user (Ctrl+C).")

    except SystemExit:
        exit_reason = "system_exit"

    except Exception as e:
        exit_reason = f"error: {type(e).__name__}"
        print(f"\n\nFatal error: {e}")
        traceback.print_exc()
        logger.error(f"Fatal error: {e}", exc_info=True)

    finally:

        # ── Clean shutdown ─────────────────────────────────────────────────────
        print("\n" + "=" * 62)
        print("  ECHORA shutting down...")
        print("=" * 62)

        if cu is not None:
            try:
                cu.shutdown()
            except Exception as e:
                print(f"  Shutdown error: {e}")

        # ── Session summary ────────────────────────────────────────────────────
        duration = time.time() - start_time
        minutes  = int(duration // 60)
        seconds  = int(duration % 60)

        print()
        print(f"  Session duration: {minutes}m {seconds}s")
        print(f"  Exit reason:      {exit_reason}")

        if cu is not None and cu._frame_count > 0:
            if cu._frame_times:
                avg_ms = sum(cu._frame_times) / len(cu._frame_times)
                fps    = 1000.0 / max(avg_ms, 1)
                print(f"  Frames processed: {cu._frame_count}")
                print(f"  Average FPS:      {fps:.1f}")
                print(f"  Slow frames:      {cu._slow_frames}")

        print()

        # ── Write session report ───────────────────────────────────────────────
        write_session_report(cu, start_time, exit_reason)

        print("=" * 62)
        print("  ECHORA stopped. Goodbye.")
        print("=" * 62)
        print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()