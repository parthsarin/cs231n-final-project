echo "Installing requirements..."
python -m pip install -r requirements.txt

export LD_LIBRARY_PATH=""

python -m wandb login

echo "Starting training..."
nohup python train.py \
    --model baseline \
    --epochs 100 \
    --lr 0.05 \
    --lambda 8 \
    --batch-size 512 \
    > train.log 2>&1 </dev/null &
