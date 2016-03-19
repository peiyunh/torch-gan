# log file 
NOW=$(date +"%Y-%m-%d-%H-%M")
LOGFILE="logs512_mpii64/$NOW.log"

# params
GPUID=0
MODELPATH="logs512_mpii64/adversarial.net"
BATCHSIZE=256

# use a common weight decay for coefL2
if [ -f "$MODELPATH" ]
then 
    th train_mpii.lua -g 0 -b $BATCHSIZE -p -n "$MODELPATH" | tee -a "$LOGFILE"
else
    th train_mpii.lua -g 0 -b $BATCHSIZE -p | tee -a "$LOGFILE"
fi
