#!/usr/bin/env python3


import os
import sys
import shlex
import argparse
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, test as serve

from watchdog.watchmedo import parser as wdparser, shell_command


# Monkey-patch SimpleHTTPRequestHandler to disable caching
old_end_headers = SimpleHTTPRequestHandler.end_headers
def end_headers_nocache(self):
    """Monkey-patch the request handler to disable browser caching."""
    self.send_header('Cache-Control', 'no-cache no-store must-revalidate')
    self.send_header('Pragma', 'no-cache')
    self.send_header('Expires', 0)
    old_end_headers(self)


def serve_sphinx(args):
    """Launch http.server to run in a background thread."""
    SimpleHTTPRequestHandler.end_headers = end_headers_nocache
    serve(HandlerClass=SimpleHTTPRequestHandler, port=args.port, bind='0.0.0.0')


def watch_sources():
    """Launch watchdog to run in a background thread, rebuilding Sphinx docs on
    file changes.
    """
    # Reuse the argparse instance from watchdog library
    args = wdparser.parse_args(args=shlex.split('shell-command --patterns="*.rst" --ignore-patterns="build/*" --recursive --command="rm -rf build/* && make html"'))
    print('Watching for changes to files matching {} (excludes {})...'.format(args.patterns, args.ignore_patterns))
    # Wait for sources to change; execute `--command` option when they do
    shell_command(args)


def main(argv):
    parser = argparse.ArgumentParser(description='''
1. Launch "python -m http.server $PORT" from the Sphinx output directory
2. Watch Sphinx docs for changes
3. Open web browser to see preview
4. Sphinx automatically rebuilds when watcher sees changes
''')
    parser.add_argument('--port', '-p', type=int, default=6789)
    args = parser.parse_args(args=argv[1:])

    if not os.path.isdir('build/html'):
        os.makedirs('build/html')

    httpserver_thread = threading.Thread(target=serve_sphinx, args=[args])
    watcher_thread = threading.Thread(target=watch_sources)

    # Run the threads, with a slight delay in between so the http.server has
    # enough time to start up (hacky, but works)
    httpserver_thread.start()
    time.sleep(1)
    watcher_thread.start()

    # Open tab in browser if not already open
    webbrowser.open('http://localhost:{}/build/html'.format(args.port), new=0)

    # Wait until threads terminate
    watcher_thread.join()
    httpserver_thread.join()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
