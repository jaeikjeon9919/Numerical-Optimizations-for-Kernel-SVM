import numpy as np



def MNIST_ten_to_binary(x_train, y_train, x_test, y_test, first_label, second_label,n):
    """
    return flatten x and y labelled with -1 and 1
    """
    x_train_flatten = []
    for i in range(len(x_train)):
        x_train_flatten.append(x_train[i].flatten())
    x_train_flatten = np.array(x_train_flatten)
    
    first_label_index_train = (y_train == first_label)
    first_label_index_train = [i for i, x in enumerate(first_label_index_train) if x]
    second_label_index_train = (y_train == second_label)
    second_label_index_train = [i for i, x in enumerate(second_label_index_train) if x]
    
    first_label_x, first_label_y = [], []
    for i in range(len(first_label_index_train)):
        first_label_x.append(x_train_flatten[first_label_index_train[i]])
        first_label_y.append(-1) # assign label -1
        
    second_label_x, second_label_y = [], []
    for i in range(len(second_label_index_train)):
        second_label_x.append(x_train_flatten[second_label_index_train[i]])
        second_label_y.append(1) # assign label 1
    
    first_label_x, first_label_y = np.array(first_label_x), np.array(first_label_y) 
    second_label_x, second_label_y = np.array(second_label_x), np.array(second_label_y)
    
    train_x = np.vstack((first_label_x, second_label_x))
    train_y = np.vstack((np.expand_dims(first_label_y, axis=1), np.expand_dims(second_label_y, axis=1)))
    
    data = np.append(train_y, train_x, axis= 1) 
    np.random.shuffle(data) # shuffle the data
    data = data[0:n] # select n data


    train_x = data[:,1:]
    train_y = data[:,0]
    
    return train_x, train_y
    
    
    
