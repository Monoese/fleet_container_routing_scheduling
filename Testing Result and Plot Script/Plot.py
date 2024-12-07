import matplotlib.pyplot as plt
import numpy as np

instance = ['1 5','1 10','1 15','1 20','1 25','2 0','2 25','2 50','2 75','2 100','3 50','3 100','3 200','3 400','3 800']
dataset = ['1_5','1_10','1_15','1_20','1_25','2_0','2_25','2_50','2_75','2_100','3_50','3_100','3_200','3_400','3_800']

lst = [[5,10,15,20,25],
       [0,25,50,75,100],
       [50,100,200,400,800]]

for n, m in zip(instance, dataset):
    globals()[f'num_of_cargo_delivered_{m}'] = np.genfromtxt(f'num of cargo delivered {n}.csv',delimiter=',')
    globals()[f'throughput_{m}'] = np.genfromtxt(f'throughput {n}.csv', delimiter=',')
    globals()[f'fuel_consumption_{m}'] = np.genfromtxt(f'theoretical and actual fuel consumption {n}.csv', delimiter=',')
    globals()[f'average_fuel_consumption_{m}'] = np.genfromtxt(f'average cost {n}.csv', delimiter=',')
    globals()[f'unit_fuel_consumption_{m}'] = np.genfromtxt(f'unit cost {n}.csv', delimiter=',')
    globals()[f'time_span_{m}'] = np.genfromtxt(f'shipping time span {n}.csv', delimiter=',')
    globals()[f'average_time_span_{m}'] = np.genfromtxt(f'average shipping time span {n}.csv', delimiter=',')
    globals()[f'unit_time_span_{m}'] = np.genfromtxt(f'unit shipping time span {n}.csv', delimiter=',')
    globals()[f'queue_size_{m}'] = np.genfromtxt(f'queue size {n}.csv', delimiter=',')
    globals()[f'num_of_ship_{m}'] = np.genfromtxt(f'num of usable ships {n}.csv', delimiter=',')

labels = [' ports in service', '% of containers routed', ' ships in service']

for n in range(len(lst)):
    fig, axes = plt.subplots(4,2,figsize=(11,8.5),constrained_layout = True)
    # plt.tight_layout()
    for m in lst[n]:
        globals()[f'queue_size_{n+1}_{m}'] = axes[2,0].plot(globals()[f'queue_size_{n+1}_{m}'][:,0],globals()[f'queue_size_{n+1}_{m}'][:,1],label = f'{m}'+labels[n])
        globals()[f'num_of_ship_model_0_{n+1}_{m}'] = axes[2,1].plot(globals()[f'num_of_ship_{n+1}_{m}'][:,0],globals()[f'num_of_ship_{n+1}_{m}'][:,1], label= f'{m}'+labels[n])
        globals()[f'num_of_ship_model_1_{n+1}_{m}'] = axes[3,0].plot(globals()[f'num_of_ship_{n+1}_{m}'][:,0],globals()[f'num_of_ship_{n+1}_{m}'][:,2], label= f'{m}'+labels[n])
        globals()[f'num_of_ship_model_2_{n+1}_{m}'] = axes[3,1].plot(globals()[f'num_of_ship_{n+1}_{m}'][:,0],globals()[f'num_of_ship_{n+1}_{m}'][:,3], label= f'{m}'+labels[n])
        globals()[f'unit_fuel_consumption_{n+1}_{m}'] = axes[1,0].plot(globals()[f'unit_fuel_consumption_{n+1}_{m}'][:,0],globals()[f'unit_fuel_consumption_{n+1}_{m}'][:,1],label = f'{m}'+labels[n])
        globals()[f'unit_time_span_{n+1}_{m}'] = axes[1,1].plot(globals()[f'unit_time_span_{n+1}_{m}'][:,0],globals()[f'unit_time_span_{n+1}_{m}'][:,1],label = f'{m}'+labels[n])
        globals()[f'fuel_consumption_{n+1}_{m}'] = axes[0,1].plot(globals()[f'fuel_consumption_{n+1}_{m}'][:,0], globals()[f'fuel_consumption_{n+1}_{m}'][:,2],label = f'{m}'+labels[n])
        globals()[f'throughput_{n+1}_{m}'] = axes[0,0].plot(globals()[f'throughput_{n+1}_{m}'][:,0], globals()[f'throughput_{n+1}_{m}'][:,1],label = f'{m}'+labels[n])

        axes[0,0].set_title("cumulative num of containers delivered")
        axes[0,0].set_xlabel("epoch")
        axes[0,0].set_ylabel("number of containers")
        # axes[0,0].set_yscale('log')

        axes[0,1].set_title("cumulative fuel consumption")
        axes[0,1].set_xlabel("epoch")
        axes[0,1].set_ylabel("fuel consumption")
        # axes[0,1].set_yscale('log')

        axes[1,0].set_title("unit cost in fuel")
        axes[1,0].set_xlabel("epoch")
        axes[1,0].set_ylabel("fuel consumption")
        # axes[1,0].set_yscale('log')

        axes[1,1].set_title("unit cost in time")
        axes[1,1].set_xlabel("epoch")
        axes[1,1].set_ylabel("time in epoch")
        # axes[1,1].set_yscale('log')

        axes[2,0].set_title("global container queue size")
        axes[2,0].set_xlabel("epoch")
        axes[2,0].set_ylabel("num of containers")
        # axes[2,0].set_yscale('log')

        axes[2,1].set_title("num of idle ships of model 0")
        axes[2,1].set_xlabel("epoch")
        axes[2,1].set_ylabel("num of ships")
        # axes[2,1].set_yscale('log')

        axes[3,0].set_title("num of idle ships of model 1")
        axes[3,0].set_xlabel("epoch")
        axes[3,0].set_ylabel("num of ships")
        # axes[3,0].set_yscale('log')

        axes[3,1].set_title("num of  ships of model 2")
        axes[3,1].set_xlabel("epoch")
        axes[3,1].set_ylabel("num of ships")
        # axes[3,1].set_yscale('log')

    axes[0,0].legend(loc="upper left")
    axes[0,1].legend(loc="upper left")
    axes[1,0].legend(loc="upper left")
    axes[1,1].legend(loc="upper left")
    axes[2,0].legend(loc="upper left")
    axes[2,1].legend(loc="upper left")
    axes[3,0].legend(loc="upper left")
    axes[3,1].legend(loc="upper left")
    plt.show()