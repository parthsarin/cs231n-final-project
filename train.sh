echo "Installing requirements..."
python -m pip install -r requirements.txt

python -m wandb login

echo "Starting training..."
python train.py \
    --model baseline \
    --epochs 100 \
    --lr 0.005 \
    --lambda 5 \
    --batch-size 512
