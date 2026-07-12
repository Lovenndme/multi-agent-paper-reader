"""Desktop launcher for the packaged Multi-Agent Paper Reader application."""

from __future__ import annotations

import ctypes
import json
import logging
import ntpath
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from ctypes import wintypes
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import BOTH, DISABLED, LEFT, NORMAL, RIGHT, Button, Frame, Label, Tk, messagebox
from typing import Any


APP_NAME = "Multi-Agent Paper Reader"
APP_VERSION = "V1.2.0"
MUTEX_NAME = "Local\\MultiAgentPaperReader.V1"
SHUTDOWN_EVENT_NAME = "Local\\MultiAgentPaperReader.Shutdown.V1"
HOST = "127.0.0.1"
PORT_RANGE = range(8000, 8011)
ERROR_ALREADY_EXISTS = 183
ERROR_INVALID_PARAMETER = 87
EVENT_MODIFY_STATE = 0x0002
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
_KERNEL32: Any | None = None


def user_root() -> Path:
    configured = os.environ.get("PAPER_READER_USER_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "MultiAgentPaperReader"


def configure_runtime() -> Path:
    root = user_root()
    data_directory = root / "data"
    logs_directory = root / "logs"
    root.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)
    logs_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PAPER_READER_ENV_PATH", str(root / ".env"))
    os.environ.setdefault("PAPER_READER_DATA_DIR", str(data_directory))
    os.environ.setdefault("PAPER_READER_VERSION", APP_VERSION)
    return root


def configure_logging(root: Path) -> None:
    handler = RotatingFileHandler(
        root / "logs" / "launcher.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def windows_kernel32() -> Any:
    """Return kernel32 with pointer-safe signatures for 64-bit Python."""
    global _KERNEL32
    if os.name != "nt":
        raise RuntimeError("Windows process controls are only available on Windows.")
    if _KERNEL32 is not None:
        return _KERNEL32

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateEventW.restype = wintypes.HANDLE
    kernel32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.OpenEventW.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.SetEvent.argtypes = [wintypes.HANDLE]
    kernel32.SetEvent.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    _KERNEL32 = kernel32
    return kernel32


def acquire_mutex() -> int | None:
    if os.name != "nt":
        return 1
    kernel32 = windows_kernel32()
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        return None
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return None
    return handle


def create_shutdown_event() -> int | None:
    if os.name != "nt":
        return 1
    return windows_kernel32().CreateEventW(None, True, False, SHUTDOWN_EVENT_NAME) or None


def resource_path(relative: str) -> Path:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    candidate = bundle_root / relative
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parent / "assets" / Path(relative).name


def state_path(root: Path) -> Path:
    return root / "runtime-state.json"


def read_runtime_state(root: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(state_path(root).read_text(encoding="utf-8"))
        pid = int(payload["pid"])
        if pid <= 0:
            return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return payload


def write_runtime_state(root: Path, port: int) -> None:
    target = state_path(root)
    temporary = target.with_suffix(".tmp")
    payload = {
        "port": port,
        "pid": os.getpid(),
        "version": APP_VERSION,
        "executable": str(Path(sys.executable).resolve()),
    }
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(target)


def remove_runtime_state(root: Path) -> None:
    try:
        state_path(root).unlink(missing_ok=True)
    except OSError:
        logging.exception("Unable to remove runtime state")


def normalized_windows_path(value: str | os.PathLike[str]) -> str:
    return ntpath.normcase(ntpath.normpath(os.fspath(value).strip('"')))


def paths_match(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    return normalized_windows_path(left) == normalized_windows_path(right)


def query_process_executable(process_handle: int) -> str | None:
    buffer = ctypes.create_unicode_buffer(32_768)
    size = wintypes.DWORD(len(buffer))
    if not windows_kernel32().QueryFullProcessImageNameW(
        process_handle,
        0,
        buffer,
        ctypes.byref(size),
    ):
        return None
    return buffer.value


def request_shutdown_for_uninstall(
    root: Path,
    *,
    graceful_timeout_ms: int = 10_000,
    forced_timeout_ms: int = 5_000,
) -> int:
    """Stop only the recorded installed process, preferring a graceful exit."""
    payload = read_runtime_state(root)
    if payload is None:
        return 0

    expected_executable = str(Path(sys.executable).resolve())
    recorded_executable = payload.get("executable")
    if recorded_executable and not paths_match(recorded_executable, expected_executable):
        logging.error("Refusing to stop PID %s: runtime executable does not match", payload["pid"])
        return 2
    if os.name != "nt":
        return 0

    kernel32 = windows_kernel32()
    access = PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE | SYNCHRONIZE
    process_handle = kernel32.OpenProcess(access, False, int(payload["pid"]))
    if not process_handle:
        if ctypes.get_last_error() == ERROR_INVALID_PARAMETER:
            remove_runtime_state(root)
            return 0
        logging.error("Unable to open recorded PID %s", payload["pid"])
        return 2

    try:
        actual_executable = query_process_executable(process_handle)
        if not actual_executable or not paths_match(actual_executable, expected_executable):
            logging.error("Refusing to stop PID %s: process identity does not match", payload["pid"])
            return 2

        event_handle = kernel32.OpenEventW(EVENT_MODIFY_STATE, False, SHUTDOWN_EVENT_NAME)
        if event_handle:
            try:
                kernel32.SetEvent(event_handle)
            finally:
                kernel32.CloseHandle(event_handle)

        wait_result = kernel32.WaitForSingleObject(process_handle, graceful_timeout_ms)
        if wait_result == WAIT_OBJECT_0:
            remove_runtime_state(root)
            return 0
        if wait_result != WAIT_TIMEOUT:
            logging.error("Unexpected wait result while stopping PID %s: %s", payload["pid"], wait_result)
            return 3

        logging.warning("Graceful shutdown timed out for PID %s; using PID-targeted termination", payload["pid"])
        if not kernel32.TerminateProcess(process_handle, 1):
            logging.error("PID-targeted termination failed for PID %s", payload["pid"])
            return 3
        if kernel32.WaitForSingleObject(process_handle, forced_timeout_ms) != WAIT_OBJECT_0:
            logging.error("PID %s did not exit after targeted termination", payload["pid"])
            return 3
        remove_runtime_state(root)
        return 0
    finally:
        kernel32.CloseHandle(process_handle)


def health_ok(port: int, timeout: float = 0.4) -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/health", timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("ok") is True
    except (OSError, ValueError, urllib.error.URLError):
        return False


def existing_url(root: Path) -> str | None:
    payload = read_runtime_state(root)
    if payload is None:
        return None
    try:
        port = int(payload["port"])
    except (ValueError, KeyError, TypeError):
        return None
    return f"http://{HOST}:{port}/" if health_ok(port) else None


def available_port() -> int:
    for port in PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError("端口 8000-8010 均被占用，请关闭相关程序后重试。")


class LauncherWindow:
    def __init__(
        self,
        runtime_root: Path,
        mutex_handle: int,
        shutdown_event_handle: int | None,
    ) -> None:
        self.runtime_root = runtime_root
        self.mutex_handle = mutex_handle
        self.shutdown_event_handle = shutdown_event_handle
        self.port = available_port()
        self.url = f"http://{HOST}:{self.port}/"
        self.server = None
        self.server_thread: threading.Thread | None = None
        self.ready_checks = 0
        self.stopping = False

        self.root = Tk()
        self.root.title(APP_NAME)
        self.root.geometry("460x270")
        self.root.resizable(False, False)
        self.root.configure(bg="#f3f7fc")
        try:
            self.root.iconbitmap(default=str(resource_path("assets/paper-reader.ico")))
        except Exception:
            logging.exception("Unable to apply the window icon")
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        header = Frame(self.root, bg="#1769e0", height=78)
        header.pack(fill="x")
        Label(
            header,
            text="Paper Reader",
            bg="#1769e0",
            fg="white",
            font=("Microsoft YaHei UI", 20, "bold"),
        ).pack(pady=(17, 0))
        Label(
            header,
            text=f"多 Agent 论文研读助手  ·  {APP_VERSION}",
            bg="#1769e0",
            fg="#dceaff",
            font=("Microsoft YaHei UI", 9),
        ).pack()

        body = Frame(self.root, bg="#f3f7fc")
        body.pack(fill=BOTH, expand=True, padx=28, pady=20)
        self.status = Label(
            body,
            text="正在启动本地服务……",
            bg="#f3f7fc",
            fg="#17324d",
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        self.status.pack(pady=(4, 5))
        self.detail = Label(
            body,
            text="首次打开浏览器后，请在 Settings 中配置自己的 GLM API Key。",
            bg="#f3f7fc",
            fg="#60758a",
            font=("Microsoft YaHei UI", 9),
        )
        self.detail.pack()

        controls = Frame(body, bg="#f3f7fc")
        controls.pack(fill="x", pady=(22, 0))
        self.open_button = Button(
            controls,
            text="打开网页",
            state=DISABLED,
            command=self.open_browser,
            bg="#1769e0",
            fg="white",
            activebackground="#0f56bd",
            activeforeground="white",
            relief="flat",
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=24,
            pady=8,
        )
        self.open_button.pack(side=LEFT)
        Button(
            controls,
            text="停止并退出",
            command=self.stop,
            bg="#e8eef6",
            fg="#2b455f",
            activebackground="#d9e3ef",
            relief="flat",
            font=("Microsoft YaHei UI", 10),
            padx=20,
            pady=8,
        ).pack(side=RIGHT)

    def start(self) -> None:
        try:
            import uvicorn
            from app import app as fastapi_app

            config = uvicorn.Config(
                fastapi_app,
                host=HOST,
                port=self.port,
                log_level="warning",
                access_log=False,
                use_colors=False,
            )
            self.server = uvicorn.Server(config)
            self.server_thread = threading.Thread(
                target=self.server.run,
                name="paper-reader-server",
                daemon=True,
            )
            self.server_thread.start()
            write_runtime_state(self.runtime_root, self.port)
            self.root.after(250, self.poll_ready)
            self.root.after(250, self.poll_shutdown_request)
            self.root.mainloop()
        except Exception as exc:
            logging.exception("Application startup failed")
            messagebox.showerror(APP_NAME, f"应用启动失败：\n{exc}")
            self.cleanup()

    def poll_ready(self) -> None:
        if health_ok(self.port):
            self.status.configure(text="网页已启动")
            self.detail.configure(text=self.url)
            self.open_button.configure(state=NORMAL)
            self.open_browser()
            return
        self.ready_checks += 1
        if self.server_thread is not None and not self.server_thread.is_alive():
            self.status.configure(text="启动失败")
            self.detail.configure(text="请查看日志或重新安装后再试。")
            messagebox.showerror(APP_NAME, "本地服务未能启动，请查看日志文件。")
            return
        if self.ready_checks >= 120:
            self.status.configure(text="启动超时")
            self.detail.configure(text="请停止后重试，或检查安全软件是否拦截。")
            return
        self.root.after(250, self.poll_ready)

    def open_browser(self) -> None:
        if os.environ.get("PAPER_READER_DISABLE_BROWSER") == "1":
            return
        webbrowser.open(self.url, new=2)

    def poll_shutdown_request(self) -> None:
        if self.stopping or not self.shutdown_event_handle or os.name != "nt":
            return
        if windows_kernel32().WaitForSingleObject(self.shutdown_event_handle, 0) == WAIT_OBJECT_0:
            logging.info("Received uninstall shutdown request")
            self.stop()
            return
        self.root.after(250, self.poll_shutdown_request)

    def stop(self) -> None:
        if self.stopping:
            return
        self.stopping = True
        self.status.configure(text="正在停止服务……")
        self.open_button.configure(state=DISABLED)
        if self.server is not None:
            self.server.should_exit = True
        self.wait_for_stop()

    def wait_for_stop(self) -> None:
        if self.server_thread is not None and self.server_thread.is_alive():
            self.root.after(100, self.wait_for_stop)
            return
        self.cleanup()
        self.root.destroy()

    def cleanup(self) -> None:
        remove_runtime_state(self.runtime_root)
        if os.name == "nt" and self.shutdown_event_handle:
            windows_kernel32().CloseHandle(self.shutdown_event_handle)
            self.shutdown_event_handle = 0
        if os.name == "nt" and self.mutex_handle:
            windows_kernel32().CloseHandle(self.mutex_handle)
            self.mutex_handle = 0


def show_existing_instance(url: str) -> None:
    root = Tk()
    root.withdraw()
    if os.environ.get("PAPER_READER_DISABLE_BROWSER") != "1":
        webbrowser.open(url, new=2)
    messagebox.showinfo(APP_NAME, "Paper Reader 已经在运行，已为你打开网页。")
    root.destroy()


def main() -> None:
    runtime_root = configure_runtime()
    configure_logging(runtime_root)
    if "--shutdown-for-uninstall" in sys.argv[1:]:
        raise SystemExit(request_shutdown_for_uninstall(runtime_root))

    mutex_handle = acquire_mutex()
    if mutex_handle is None:
        url = existing_url(runtime_root)
        if url:
            show_existing_instance(url)
        else:
            messagebox.showwarning(APP_NAME, "Paper Reader 正在启动，请稍后再试。")
        return
    shutdown_event_handle = create_shutdown_event()
    if shutdown_event_handle is None:
        logging.warning("Unable to create graceful shutdown event; PID-targeted fallback remains available")
    LauncherWindow(runtime_root, mutex_handle, shutdown_event_handle).start()


if __name__ == "__main__":
    main()
