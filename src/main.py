
import sys
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x00000080
    )

import argparse
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime

def parse_args():

    parser = argparse.ArgumentParser(
        description="ECHORA — AI-Powered Sensory Substitution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                   Auto mode — state machine controls everything
  python main.py --manual          Manual mode — keyboard controls mode
  python main.py --no-display      Wearable mode — no screen needed
  python main.py --debug           Verbose logging
  python main.py --log-file        Save session log to logs/
  python main.py --manual --debug  Manual mode with full debug logging
        """
    )

    parser.add_argument(
        "--manual",
        action = "store_true",
        help   = (
            "Start in MANUAL testing mode. "
            "Press TAB to toggle auto/manual, 1-5 to select mode. "
            "Emergency override is disabled in manual mode."
        )
    )
    parser.add_argument(
        "--debug",
        action = "store_true",
        help   = "Enable verbose DEBUG logging"
    )
    parser.add_argument(
        "--no-display",
        action = "store_true",
        dest   = "no_display",
        help   = "Disable debug window — headless wearable mode"
    )
    parser.add_argument(
        "--no-audio",
        action = "store_true",
        dest   = "no_audio",
        help   = "Disable all audio output"
    )
    parser.add_argument(
        "--log-file",
        action = "store_true",
        dest   = "log_file",
        help   = "Save log to logs/session_TIMESTAMP.log"
    )
    parser.add_argument(
        "--tolerance",
        type    = float,
        default = None,
        help    = "Face recognition tolerance 0.0-1.0"
    )

    return parser.parse_args()

def setup_logging(debug: bool = False, log_file: bool = False):

    level         = logging.DEBUG if debug else logging.INFO
    echora_logger = logging.getLogger("ECHORA")
    echora_logger.setLevel(level)
    echora_logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)

    if hasattr(console.stream, 'reconfigure'):
        try:
            console.stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

    console.setFormatter(logging.Formatter(
        fmt     = "%(asctime)s — %(levelname)s — %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    ))
    echora_logger.addHandler(console)

    if log_file:
        logs_dir  = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path  = logs_dir / f"session_{timestamp}.log"

        fh = logging.FileHandler(str(log_path), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            fmt     = "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S"
        ))
        echora_logger.addHandler(fh)
        print(f"  Log file: {log_path}")

    return echora_logger

def print_banner(manual_mode: bool):

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
    print("=" * 62)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if manual_mode:
        print("  Mode:    MANUAL TESTING")
        print()
        print("  Controls:")
        print("    TAB = toggle AUTO / MANUAL")
        print("    1   = NAVIGATION")
        print("    2   = OCR")
        print("    3   = INTERACTION")
        print("    4   = FACE_ID")
        print("    5   = BANKNOTE")
        print("    Q   = quit")
    else:
        print("  Mode:    AUTO (state machine controls everything)")
        print()
        print("  Controls:")
        print("    TAB = switch to MANUAL testing mode")
        print("    Q   = quit")

    print("=" * 62)
    print()

def write_session_report(cu, start_time: float, exit_reason: str):

    try:
        logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = logs_dir / f"report_{timestamp}.txt"

        duration = time.time() - start_time
        h = int(duration // 3600)
        m = int((duration % 3600) // 60)
        s = int(duration % 60)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("ECHORA Session Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date:        {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"Start:       {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}\n")
            f.write(f"Duration:    {h:02d}:{m:02d}:{s:02d}\n")
            f.write(f"Exit reason: {exit_reason}\n\n")

            if cu is not None and cu._started:
                if cu._frame_times:
                    avg_ms = sum(cu._frame_times) / len(cu._frame_times)
                    fps    = 1000.0 / max(avg_ms, 1)
                else:
                    avg_ms = fps = 0

                f.write("Performance\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total frames: {cu._frame_count}\n")
                f.write(f"Average FPS:  {fps:.1f}\n")
                f.write(f"Slow frames:  {cu._slow_frames}\n\n")

                if cu._state_machine:
                    sm = cu._state_machine.get_stats()
                    f.write("State Machine\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Mode switches: {sm.get('total_switches', 0)}\n")
                    f.write(f"Final mode:    {sm.get('current_mode', 'N/A')}\n\n")

                try:
                    from src.storage.database import get_db
                    db = get_db()
                    if db:
                        s = db.get_stats()
                        f.write("Database\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"Persons: {s.get('persons', 0)}\n")
                        f.write(f"Events:  {s.get('events',  0)}\n\n")
                except Exception:
                    pass
            else:
                f.write("System did not start successfully.\n")

        print(f"\n  Session report: {report_path}")

    except Exception as e:
        print(f"\n  Could not write session report: {e}")

def main():

    args = parse_args()

    print_banner(manual_mode=args.manual)

    logger = setup_logging(debug=args.debug, log_file=args.log_file)

    if args.no_display:
        import src.core.control_unit as cu_module
        cu_module.SHOW_DEBUG_WINDOW = False
        print("  Display:   OFF (headless)")
    else:
        print("  Display:   ON")

    if args.tolerance is not None:
        from src.core.config import settings
        settings.FACE_RECOGNITION_TOLERANCE = args.tolerance
        print(f"  Tolerance: {args.tolerance}")

    if args.debug:
        print("  Logging:   DEBUG")

    if args.no_audio:
        print("  Audio:     OFF")

    print()

    start_time  = time.time()
    exit_reason = "unknown"
    cu          = None

    try:
        from src.core.control_unit import ControlUnit

        cu = ControlUnit(start_in_manual=args.manual)

        if args.no_audio:
            class _SilentAudio:
                _ready = True
                def speak(self, *a, **kw):                pass
                def announce_obstacle(self, *a, **kw):    pass
                def announce_scene(self, *a, **kw):       pass
                def announce_ocr(self, *a, **kw):         pass
                def announce_face(self, *a, **kw):        pass
                def announce_banknote(self, *a, **kw):    pass
                def announce_mode_change(self, *a, **kw): pass
                def play_spatial_alert(self, *a, **kw):   pass
                def stop_all(self, *a, **kw):              pass
                def release(self):                        pass
            _silence = True
        else:
            _silence = False

        cu.startup()

        if _silence:
            cu._audio = _SilentAudio()

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

        print("\n" + "=" * 62)
        print("  ECHORA shutting down...")
        print("=" * 62)

        if cu is not None and not cu._started:
            try:
                cu.shutdown()
            except Exception:
                pass

        duration = time.time() - start_time
        m = int(duration // 60)
        s = int(duration % 60)

        print()
        print(f"  Duration: {m}m {s}s")
        print(f"  Exit:     {exit_reason}")

        if cu is not None and cu._frame_count > 0 and cu._frame_times:
            avg_ms = sum(cu._frame_times) / len(cu._frame_times)
            fps    = 1000.0 / max(avg_ms, 1)
            print(f"  Frames:   {cu._frame_count}")
            print(f"  Avg FPS:  {fps:.1f}")
            print(f"  Slow:     {cu._slow_frames}")

        print()
        write_session_report(cu, start_time, exit_reason)

        print("=" * 62)
        print("  ECHORA stopped. Goodbye.")
        print("=" * 62)
        print()

if __name__ == "__main__":
    main()