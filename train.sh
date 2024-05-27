echo "Installing requirements..."
python -m pip install -r requirements.txt

echo "Starting training..."
python train.py \
    --model baseline \
    --epochs 100 \
    --learning_rate 0.05 \
    --batch_size 64
