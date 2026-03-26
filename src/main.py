
# instalacja zależności
# python -m pip install -r requirements.txt
# python -m pip install -r requirements-torch-cu130.txt

# split danych
# python scripts/make_split.py --src data/raw/RVL-CDIP --dst data/processed/RVL-CDIP_subset --train 800 --val 100 --test 100 --hardlinks

#SimpleCNN
    # trening
    # 3 epoki
    # python -m src.training.train --data_dir data/processed/RVL-CDIP_subset --epochs 3 --batch_size 32 --img_size 224 --num_workers 4
    # 10 epoków
    # python -m src.training.train --data_dir data/processed/RVL-CDIP_subset --epochs 10 --batch_size 64 --img_size 224 --num_workers 8
    # 20 epok
    # python -m src.training.train --data_dir data/processed/RVL-CDIP_subset --epochs 20 --batch_size 64 --img_size 224 --num_workers 8
    # 30 epok
    # python -m src.training.train --data_dir data/processed/RVL-CDIP_subset --epochs 30 --batch_size 64 --img_size 224 --num_workers 8

    # test - evaluate
    # python -m src.training.evaluate --data_dir data/processed/RVL-CDIP_subset --ckpt outputs/checkpoints/best.pt --batch_size 32 --num_workers 4
    # python -m src.training.evaluate --data_dir data/processed/RVL-CDIP_subset --ckpt outputs/checkpoints/20260118-225520/best.pt --batch_size 64 --num_workers 8
    # python -m src.training.evaluate --data_dir data/processed/RVL-CDIP_subset --ckpt outputs/checkpoints/20260119-195951/best.pt --batch_size 64 --num_workers 8
    # python -m src.training.evaluate --data_dir data/processed/RVL-CDIP_subset --ckpt outputs/checkpoints/20260119-213309/best.pt --batch_size 64 --num_workers 8

# tensorboard
# tensorboard --logdir outputs\checkpoints --port 6006
# albo python -m tensorboard.main --logdir outputs\checkpoints --port 6006
# http://localhost:6006

#VIT
    # vit 1 epoka dla sprawdzenia
    # python -m src.training.train --data_dir data/processed/RVL-CDIP_subset --model vit --pretrained --epochs 1 --batch_size 32 --img_size 224 --num_workers 8 --lr 0.0003

    # puszczenie od tak
    # python -m src.training.train --data_dir data/processed/RVL-CDIP_subset --model vit --pretrained --epochs 20 --batch_size 64 --img_size 224 --num_workers 8 --lr 0.0001
    #nowa z run_name na końcu
    #python -m src.training.train --data_dir data/processed/RVL-CDIP_subset --model vit --pretrained --epochs 20 --batch_size 64 --num_workers 8 --lr 0.0001 --run_name lr1e-4_bs64

    # python -m src.training.evaluate --data_dir data/processed/RVL-CDIP_subset --ckpt outputs/checkpoints/20260305-204545-483/best.pt --batch_size 64 --num_workers 8