# Visualize training history
import matplotlib.pyplot as plt
import file_utils


def history(train_accs, val_accs, name):

    plot_name = name + '.png'

    x_axis = range(1, len(train_accs)+1)

    plt.plot(train_accs, x_axis)
    plt.plot(val_accs, x_axis)
    plt.title("Dice's coefficient by epoch")
    plt.ylabel("Dice's coefficient")
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(plot_name)
    plt.show()


def history_fedavg(train_accs, val_accs, clientsnbr, name):

    plot_name = name + '.png'
    colors = ['blue', 'red', 'green']

    x_axis = range(1, len(train_accs) + 1)

    for i in range(clientsnbr):
        train_accs_client = []
        for j in range(i, len(train_accs), clientsnbr):
            train_accs_client.append(train_accs[j])
        print(train_accs_client)
        plt.plot(train_accs_client, x_axis, color=colors[i])
    for i in range(clientsnbr):
        val_accs_client = []
        for j in range(i, len(val_accs), clientsnbr):
            val_accs_client.append(val_accs[j])
        print(val_accs_client)
        plt.plot(val_accs_client, x_axis, color=colors[i], linestyle='dotted')
    plt.title("Dice's coefficient by epoch")
    plt.ylabel("Dice's coefficient")
    plt.xlabel('epoch')
    plt.legend(['train client 1', 'train client 2', 'train client 3', 'validation client 1', 'validation client 2', 'validation client 3'], loc='lower right')
    plt.savefig(plot_name)
    plt.show()


def plot_from_file(filepath, name):
    plot_name = name + '.png'

    lines = file_utils.read_measures(filepath)
    mid = int(len(lines)/2)
    print("epoch optimal: " + str(mid-10))
    train = lines[:mid]
    val = lines[mid:]
    history(train, val, plot_name)


def interpret_fed_path(fed_path, dataset_nbr, nbclients):
    lines = file_utils.read_measures(fed_path)
    separation = int(len(lines)/7)
    train = lines[0:separation*3] # Train acc of the three clients
    train_client = []

    # Takes only the train of the interesting dataset
    for i in range(dataset_nbr, len(train), nbclients):
        train_client.append(train[i])

    val = lines[separation*3:separation*6]
    val_client = []

    # Takes only the train of the interesting dataset
    for i in range(dataset_nbr, len(val), nbclients):
        val_client.append(val[i])

    return train_client, val_client



def compare_fedAvg_to_separate_models(local_path, fed_path, client_nbr, nbclients, name):
    plot_name = name + '.png'

    local_lines = file_utils.read_measures(local_path)
    local_mid = int(len(local_lines) / 2)
    local_train = local_lines[:local_mid]
    local_val = local_lines[local_mid:]

    x_axis1 = range(1, len(local_train) + 1)

    plt.plot(local_train, x_axis1, color='blue')
    plt.plot(local_val, x_axis1, color='blue', linestyle='dotted')

    fed_train, fed_val = interpret_fed_path(fed_path, client_nbr, nbclients)

    x_axis2 = range(1, len(fed_train) + 1)

    plt.plot(fed_train, x_axis2, color='red')
    plt.plot(fed_val, x_axis2, color='red', linestyle='dotted')

    plt.title("FedAvg vs local training for dataset " + str(client_nbr+1))
    plt.ylabel("Dice's coefficient")
    plt.xlabel('epoch')
    plt.legend(['local train', 'local validation', 'fedAvg train', 'fedAvg validation'], loc='lower right')

    plt.savefig(plot_name)
    plt.show()
