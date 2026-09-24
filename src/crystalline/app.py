"""QApplication bootstrap."""

from __future__ import annotations

import os
import sys
from typing import Optional

from crystalline.core.structure import Structure

_APP_NAME = "CRYSTALLine"


def run(structure: Optional[Structure] = None) -> int:
    """Launch the CRYSTALLine GUI. Returns the Qt exit code.

    The order here is the startup time. Importing the main window pulls in VTK,
    PyVista and Qt's 3D interactor — about four seconds — and on a machine that
    has never run a matplotlib program, another ten while matplotlib scans every
    font installed and caches the result. Done before ``QApplication`` exists,
    as it used to be, that is quarter of a minute of nothing at all: no window,
    no dock bounce, nothing to say the program is starting.

    So Qt comes up first and puts the logo on screen, and the heavy import
    happens behind it.
    """
    # Imported here so `import crystalline.app` doesn't require a display. Note
    # what is *not* imported yet: the main window, and everything under it.
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from crystalline.resources import logo_path
    from crystalline.ui import safety, theme

    # Both must run *before* QApplication is created: Qt reads the macOS bundle
    # name once, at construction, and the platform plugin is chosen there too.
    _name_macos_app(_APP_NAME)
    _prefer_x11_on_wayland()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(_APP_NAME)
    app.setApplicationDisplayName(_APP_NAME)  # window-title suffix on some platforms
    app.setDesktopFileName(_APP_NAME)  # X11/Wayland app id
    app.setWindowIcon(QIcon(logo_path()))  # dock / taskbar icon
    # Before anything that can fail: an error in a Qt virtual override takes the
    # process down with it (shiboken segfaults reporting it), so the net has to
    # be up before the first window exists.
    safety.install()
    theme.apply(app)  # the app's own look, light or dark to match the desktop

    splash = _splash(app)
    warming = _warm_font_cache()

    from crystalline.ui.main_window import MainWindow  # the slow part

    window = MainWindow(structure)
    if warming is not None:
        warming.join()  # the window is about to draw labels with those fonts
    window.show()
    if splash is not None:
        splash.finish(window)
    return app.exec()


def _splash(app):
    """Put the logo on screen while the main window is being built.

    Returns the splash screen, or ``None`` if it could not be shown — in which
    case startup simply carries on without it. The message says which of the two
    waits this is, because a first run takes three times as long as the rest and
    looks, without a word, like the program has hung.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QPixmap
        from PySide6.QtWidgets import QSplashScreen

        from crystalline.resources import logo_path
        from crystalline.ui import theme

        pixmap = QPixmap(logo_path())
        if pixmap.isNull():
            return None
        pixmap = pixmap.scaledToWidth(360, Qt.SmoothTransformation)
        splash = QSplashScreen(pixmap)
        message = "Starting…" if _fonts_are_cached() else "Starting — preparing fonts, first run only…"
        colour = QColor(theme.active_palette(app).text)
        splash.showMessage(message, Qt.AlignHCenter | Qt.AlignBottom, colour)
        splash.show()
        app.processEvents()  # nothing is painted until the event loop is let run
        return splash
    except Exception:  # noqa: BLE001 - a missing splash must never stop the app
        return None


def _fonts_are_cached() -> bool:
    """Whether matplotlib has already scanned this machine's fonts.

    Importing ``matplotlib`` is cheap (about a tenth of a second) and does *not*
    build the cache; only ``font_manager`` does. So the cache can be asked about
    before anything pays for it.
    """
    try:
        import pathlib

        import matplotlib

        cache = pathlib.Path(matplotlib.get_cachedir())
        return any(cache.glob("fontlist-v*.json"))
    except Exception:  # noqa: BLE001 - assume cached; the message is a nicety
        return True


def _warm_font_cache():
    """Start matplotlib's font scan on a background thread. Returns the thread.

    ``pyvistaqt`` imports ``matplotlib.font_manager``, which on a machine with no
    cache scans every installed font — eleven seconds here. That happens partway
    through the main window's import either way; starting it now lets it run
    while VTK and Qt are still loading, instead of after them. The main thread
    blocks on the module's own import lock when it gets there, by which time most
    of the scan is done.
    """
    import threading

    def build() -> None:
        try:
            import matplotlib.font_manager  # noqa: F401 - imported for its side effect
        except Exception:  # noqa: BLE001 - it will be imported again, and report then
            pass

    thread = threading.Thread(target=build, name="font-cache", daemon=True)
    thread.start()
    return thread


def _prefer_x11_on_wayland() -> bool:
    """On a Wayland session, ask Qt for the X11 (xcb) plugin. Returns whether set.

    VTK's render window is an X11 one: pyvistaqt hands it the Qt widget's
    ``winId()`` as a string, which ``vtkXOpenGLRenderWindow::SetWindowInfo``
    parses into an X ``Window`` id. Under a native Wayland Qt plugin that id is
    a 64-bit surface pointer, so the parse overflows ("The result is out of
    range, failed to get the converted tmp") and VTK then calls
    ``XConfigureWindow`` on a bogus id — Xlib aborts the process with
    ``BadWindow``. Running Qt on XWayland instead gives a real X window id.

    Only applied when the user hasn't chosen a platform plugin themselves and
    an X display (XWayland) is actually there to fall back to.
    """
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("QT_QPA_PLATFORM"):
        return False  # an explicit choice is the user's to make
    on_wayland = (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        or bool(os.environ.get("WAYLAND_DISPLAY"))
    )
    if not on_wayland or not os.environ.get("DISPLAY"):
        return False
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    return True


def _name_macos_app(name: str) -> None:
    """Make the macOS menu bar read ``name`` instead of "Python".

    A non-bundled Python app inherits the interpreter's name in the application
    menu (the bold item next to the Apple menu). Qt derives that title from the
    main bundle's ``CFBundleName``, so we set it via the Objective-C runtime —
    through ``ctypes`` so no extra dependency (pyobjc) is required. A no-op off
    macOS, or if anything is unavailable; never fatal.
    """
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        # Foundation defines NSBundle/NSString; load it so those classes exist
        # even before Qt (which also pulls it in) has initialised.
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        def send(restype, argtypes, receiver, selector, *args):
            objc.objc_msgSend.restype = restype
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
            return objc.objc_msgSend(receiver, objc.sel_registerName(selector), *args)

        main_bundle = send(ctypes.c_void_p, [], objc.objc_getClass(b"NSBundle"), b"mainBundle")
        info = send(ctypes.c_void_p, [], main_bundle, b"infoDictionary")
        if not info:
            return  # a bundle-less context we can't rename — leave it be

        ns_string = objc.objc_getClass(b"NSString")

        def nsstr(text: str):
            return send(
                ctypes.c_void_p, [ctypes.c_char_p], ns_string,
                b"stringWithUTF8String:", text.encode("utf-8"),
            )

        send(
            ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p], info,
            b"setObject:forKey:", nsstr(name), nsstr("CFBundleName"),
        )
    except Exception:  # noqa: BLE001 - a cosmetic rename must never block startup
        pass


__all__ = ["run"]
