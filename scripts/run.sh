ulimit -n 40960
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

n_client_epochs=5
early_stopping=0
n_class1=62
n_class2=10
n_class3=150
n_clients=6
rounds=100
frac=0.4
mu=0.5
max_cluster=4

is_testing=1

for non_iid in 1
do

for aggregator in fedavg fedprox fednova
do

    python -u ../run.py \
    --aggregator $aggregator \
    --n_clients $n_clients \
    --frac $frac \
    --rounds $rounds \
    --n_client_epochs $n_client_epochs \
    --data cifar10 \
    --n_class $n_class2 \
    --optim sgd \
    --max_cluster $max_cluster \
    --lr 0.01 \
    --port 50052 \
    --momentum 0.9 \
    --early_stopping $early_stopping \
    --use_gpu \
    --is_testing $is_testing \
    --clients_per_rounds 1 \
    --non_iid $non_iid \
    --gpu 2 \
    --mu $mu \
    --n_cluster 2 \
    --batch_size 600 \
    --use_multiple_gpu

done

for aggregator in fedavg fedprox fednova
do
    python -u ../run.py \
    --model_name cnn \
    --data_root ../datasets/ \
    --data cifar100 \
    --aggregator $aggregator \
    --n_clients $n_clients \
    --frac $frac \
    --rounds $rounds \
    --n_client_epochs $n_client_epochs \
    --n_class $n_class3 \
    --max_cluster $max_cluster \
    --optim sgd \
    --lr 0.01 \
    --port 50052 \
    --momentum 0.9 \
    --early_stopping $early_stopping \
    --use_gpu \
    --is_testing $is_testing \
    --clients_per_rounds 1 \
    --non_iid $non_iid \
    --gpu 2 \
    --mu $mu \
    --n_cluster 2 \
    --batch_size 600 \
    --use_multiple_gpu

done

for aggregator in fedavg fedprox fednova
do
python -u ../run.py \
    --model_name cnn \
    --data_root ../datasets/ \
    --data FashionMNIST \
    --n_clients $n_clients \
    --aggregator $aggregator \
    --frac $frac \
    --rounds $rounds \
    --n_client_epochs $n_client_epochs \
    --n_class 10 \
    --max_cluster $max_cluster \
    --optim sgd \
    --lr 0.01 \
    --port 50052 \
    --momentum 0.9 \
    --early_stopping $early_stopping \
    --use_gpu \
    --is_testing $is_testing \
    --clients_per_rounds 1 \
    --non_iid $non_iid \
    --gpu 2 \
    --mu $mu \
    --n_cluster 2 \
    --batch_size 600 \
    --use_multiple_gpu
done

for aggregator in fedavg fedprox fednova
do

    python -u ../run.py \
    --model_name cnn \
    --data_root ../datasets/ \
    --data EMNIST \
    --n_clients $n_clients \
    --aggregator $aggregator \
    --frac $frac \
    --rounds $rounds \
    --n_client_epochs $n_client_epochs \
    --n_class $n_class1 \
    --max_cluster $max_cluster \
    --optim sgd \
    --lr 0.01 \
    --port 50052 \
    --momentum 0.9 \
    --early_stopping $early_stopping \
    --use_gpu \
    --is_testing $is_testing \
    --clients_per_rounds 1 \
    --non_iid $non_iid \
    --gpu 2 \
    --mu $mu \
    --n_cluster 2 \
    --batch_size 600 \
    --use_multiple_gpu
done

done
