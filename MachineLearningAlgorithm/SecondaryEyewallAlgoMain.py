from BinaryMultiOutput import BinaryMultiOutputClass
from ModBinaryMultiOutput import ModifiedBinaryMultiOutputClass

fp = "/Users/mitchelhuott/Downloads/IotaEyewallDataset.csv"
fp1 = r'/Volumes/MHUOTT_TC/Hurricane Research/Tropical Cylone/CompleteDatasets/GenevieveEyewallDataset.csv'


user = BinaryMultiOutputClass(fp)
train_loader, test_loader = user.dataload()
model = user.model()
user.train(model, train_loader, test_loader)
'''user.evaluate(model, test_loader)'''

print("GeneveiveEyewallDataset.csv")
Genevieve = BinaryMultiOutputClass(fp1)
train_loader1, test_loader1 = Genevieve.dataload()
Genevieve.evaluate(model, test_loader1, verbose=True)
