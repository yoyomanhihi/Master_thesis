# Visualize training history
import matplotlib.pyplot as plt


def history(history, name):

    plot_name = name + '.png'

    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title("Dice's coefficient by epoch")
    plt.ylabel("Dice's coefficient")
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(plot_name)
    plt.show()


def history_fedavg(train_accs, val_accs, clientsnbr, name):

    plot_name = name + '.png'

    for i in range(clientsnbr):
        train_accs_client = []
        for j in range(i, len(train_accs), clientsnbr):
            train_accs_client.append(train_accs[j])
        print(train_accs_client)
        plt.plot(train_accs_client)
    for i in range(clientsnbr):
        val_accs_client = []
        for j in range(i, len(val_accs), clientsnbr):
            val_accs_client.append(val_accs[j])
        print(val_accs_client)
        plt.plot(val_accs_client)
    plt.title('dice score by epoch')
    plt.ylabel('dice score')
    plt.xlabel('epoch')
    plt.legend(['train client 1', 'train client 2', 'train client 3', 'validation client 1', 'validation client 2', 'validation client 3'], loc='lower right')
    plt.savefig(plot_name)
    plt.show()
