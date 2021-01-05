# define global variables
NAME=$1
WANDB="./src/wandb/"
RUNS="./src/runs/"
CACHE="./src/__pycache__/"
DATE=$(date +%F-%H-%M-%S)
RESULT="./src/result/"
OUTS="./outs"

usage() {
	echo "Usage: $0 NAME"
	echo "---------------------------------------------------------------"
	echo "> NAME   |   specify the name of backup result file"
	echo "---------------------------------------------------------------"
	echo "example: $ sh ./clean.sh NAME"
	echo "1. remove cache directories in ./src/"
      echo "2. backup ./src/result as ./outs/NAME-results.tar"
	echo "---------------------------------------------------------------"
	exit 2
}

if [ "$NAME" = "help" ]; then
      usage
fi

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
      if [ ! -d "$OUTS" ]; then
            mkdir $OUTS
      fi
      if [ -z "$NAME" ]; then
            tar -cvf ${DATE}-results.tar $RESULT
            mv ${DATE}-results.tar $OUTS
      else
            tar -cvf ${NAME}-results.tar $RESULT
            mv ${NAME}-results.tar $OUTS
      fi
      rm -rf $RESULT
fi