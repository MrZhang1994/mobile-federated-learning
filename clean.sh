# define global variables
NAME=$1
WANDB="./src/wandb/"
CACHE="./src/__pycache__/"
CACHE2="./src/utils/__pycache__/"
DATE=$(date +%F-%H-%M-%S)
RESULT="./src/result/"
TMP="./src/tmp/"
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

# remove __pycache dir
if [ -d "$CACHE" ]; then
      rm -rf $CACHE
fi

# remove utils/__pycache__ dir
if [ -d "$CACHE2" ]; then
      rm -rf $CACHE2
fi

if [ -d "$TMP" ]; then
      rm -rf $TMP
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