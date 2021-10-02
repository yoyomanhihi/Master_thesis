import medical_FL_utils as med_utils

# datasetpath1 = '2d_dataset_1.pickle'
datasetpath2 = 'small_2d_dataset_1.pickle'
datasetpath3 = 'small_2d_dataset_2.pickle'
listdatasetspaths = [datasetpath2, datasetpath3]

clients, x_test, y_test = med_utils.createClients(listdatasetspaths)

model = med_utils.fedAvg(clients, x_test, y_test)

model.save('fedAvg_model.h5')