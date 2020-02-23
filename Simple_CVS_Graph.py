import csv
import re
import matplotlib.pyplot as plt


with open("Simple_Final_75_75.csv") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    data = [r for r in reader]

epoch = []
train_accuracy = []
train_loss = []
validation_accuracy = []
validation_loss = []

for d in data:

    d = str(d).strip("[']'=")

    # new = re.sub("][", "", d)
    # print(new)
    newstr = d.replace("[']", "")

    a_split = str(d).split(';')
    epoch.append(int(a_split[0]))
    train_accuracy.append(float(a_split[1]))
    train_loss.append(float(a_split[2]))
    validation_accuracy.append(float(a_split[3]))
    validation_loss.append(float(a_split[4]))


print(epoch)
print(train_accuracy)
print(train_loss)

f = plt.figure(1)
# plotting the line 1 points
plt.plot(epoch, train_accuracy, label="Train_Accuracy")

# plotting the line 2 points
plt.plot(epoch, train_loss, label="Train_Loss")

# naming the x axis
plt.xlabel('Number of Epochs')
# naming the y axis
plt.ylabel('Percentage')
# giving a title to my graph
plt.title('Epoch Training Accuracy & Loss')

# show a legend on the plot
plt.legend()

# function to show the plot

f2 = plt.figure(2)
# plotting the line 1 points
plt.plot(epoch, validation_accuracy, label="Validation_Accuracy")

# plotting the line 2 points
plt.plot(epoch, validation_loss, label="Validation_Loss")

# naming the x axis
plt.xlabel('Number of Epochs')
# naming the y axis
plt.ylabel('Percentage')
# giving a title to my graph
plt.title('Epoch Validation Accuracy & Loss')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()