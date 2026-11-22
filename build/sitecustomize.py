"""Installed into the site-packages of the added interpreter build (see fetch_python.py).

That build has the sqlite3 extension compiled in, with its own, newer SQLite library. Tests
that execute SQL in sqlite3 must see the SQLite of the pinned image (the one the labels were
made with), so "import _sqlite3" is answered with the pinned image's own extension module
file, which is linked against the image's libsqlite3. The file is built for the same
CPython version and ABI and binds to the running interpreter's symbols (it does not link a
libpython). check_python.py proves all of this at image build time."""
import sys

OFFICIAL_SQLITE = "/usr/local/lib/python3.13/lib-dynload/_sqlite3.cpython-313-x86_64-linux-gnu.so"


class ImageSqliteFinder:
    """Import finder that is asked before the finder of compiled-in modules."""

    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name != "_sqlite3":
            return None
        from importlib.machinery import ExtensionFileLoader, ModuleSpec

        spec = ModuleSpec(name, ExtensionFileLoader(name, OFFICIAL_SQLITE), origin=OFFICIAL_SQLITE)
        spec.has_location = True  # the module gets __file__, as in the pinned image
        return spec

    @classmethod
    def invalidate_caches(cls):
        pass


sys.meta_path.insert(0, ImageSqliteFinder)
