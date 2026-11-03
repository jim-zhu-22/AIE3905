#!/bin/bash
cd -- "$(dirname -- "$0")"
exec /bin/bash "./start_annotation_demo.command" --install-only "$@"
