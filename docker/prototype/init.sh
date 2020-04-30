#!/bin/bash
nohup jupyter-lab --allow-root --ip=0.0.0.0 --port=$JUPYTERLAB_PORT --no-browser --NotebookApp.token='' > $LOGS_DIR/jupyter-lab.log 2>&1 &
nohup code-server --bind-addr 0.0.0.0:$CODESERVER_PORT --auth none > $LOGS_DIR/code-server.log dev.code-workspace 2>&1 &
echo "IDEs started (jupyter-lab:$JUPYTERLAB_PORT, coder-server:$CODESERVER_PORT)"