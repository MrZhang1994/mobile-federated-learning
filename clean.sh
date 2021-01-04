# define global variables
WANDB="./src/wandb/"
RUNS="./src/runs/"
CACHE="./src/__pycache__/"
DATE=$(date +%F-%H-%M-%S)
RESULT="./src/result/"
OUTS="./outs"

# remove wandb dir 
if [ -d "$WANDB" ]; then
rm -rf $WANDB
fi

# remove runs dir
if [ -d "$RUNS" ]; then
rm -rf $RUNS
fi

# remove __pycache__ dir
if [ -d "$CACHE" ]; then
rm -rf $CACHE
fi

if [ -d "$RESULT" ]; then
      tar -cvf ${DATE}-results.tar $RESULT
      rm -rf $RESULT
      if [ ! -d "$OUTS" ]; then
            mkdir $OUTS
      fi
      mv ${DATE}-results.tar $OUTS
fi