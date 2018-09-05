import N_Network, DSgen, model, tf_utils

def main():


    DSgen.image_encode() ##encode image into file
    x_train, y = DSgen.load_data() ##read the file and decode the file

    y_train = tf_utils.one_hot_matrix(y, 2) ##(y, # classes)
    x_test = x_train
    y_test = y_train

    parameters = model.train_model(x_train, y_train, x_test, y_test)

    N_Network.prediction(parameters, "1.jpg")


main()