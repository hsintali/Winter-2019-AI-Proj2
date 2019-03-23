import math
import matplotlib.pyplot as plt
import time
path = ""
def euclidean(trainSet, testSet, selectedFeatures):
    dist = 0
    for i in selectedFeatures:
        dist += (trainSet[i] - testSet[i])**2
    return math.sqrt(dist)

def nearestNeighbor(trainSet, testData, selectedFeatures):
    distance = []
    for i in trainSet:
        dist = euclidean(i, testData, selectedFeatures)
        distance.append([dist, i[0]])
    distance = sorted(distance)
    return distance[0][1]

def crossValidation(data, selectedFeatures, leastError = -1):
    correct = 0
    error = 0
    features = selectedFeatures
    for i in range(0, len(data)):
        train = data[:]
        test = train.pop(i)
        p = nearestNeighbor(train, test, features)
        if p == test[0]:
            correct += 1
        elif leastError != -1:
            error += 1
            if error > leastError:
                return 0, leastError
    if leastError == -1:
         return float(correct) / float(len(data)) * 100
    else:
        leastError = error
        return float(correct) / float(len(data)) * 100, leastError

def forwardSelection(data, excludeFeatures, mode = 'normal'):
    start = time.time()
    selectedFeatures = []
    finalFeatures = []
    finalAccuracy = 0
    drawAccuracy = []
    drawFeatures = [0]
    leastError = len(data)

    if mode == 'faster':
        defaultRate, n = crossValidation(data, [], leastError)
    elif mode == 'normal':
        defaultRate = crossValidation(data, [])

    if defaultRate < 50:
        drawAccuracy.append(100 - defaultRate)
    else:
        drawAccuracy.append(defaultRate)

    for i in range(1, len(data[0])):
        bestAccuracy = 0
        pendingFeature = 0
        for j in range(1, len(data[0])):
            if j not in selectedFeatures and j not in excludeFeatures:
                selectedFeatures.append(j)
                if mode == 'faster':
                    accuracy, leastError = crossValidation(data, selectedFeatures, leastError)
                elif mode == 'normal':
                    accuracy = crossValidation(data, selectedFeatures)
                print("        Using feature(s) " + str(selectedFeatures) + " accuracy is " + str(accuracy) + "%")
                selectedFeatures.remove(j)
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    pendingFeature = j

        if pendingFeature == 0:
            break
        elif pendingFeature != 0:
            selectedFeatures.append(pendingFeature)
            drawFeatures.append(pendingFeature)
            drawAccuracy.append(bestAccuracy)

        if bestAccuracy > finalAccuracy:
            finalAccuracy = bestAccuracy
            finalFeatures = selectedFeatures[:]
            print("\nFeature set " + str(selectedFeatures) + " was best, accuracy is " + str(bestAccuracy) + "%\n")
        else:
            print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            print("Feature set " + str(selectedFeatures) + " was best, accuracy is " + str(bestAccuracy) + "%\n")

    print("\nFinished search!! The best feature subset is " + str(finalFeatures) + ", which has an accuracy of " + str(finalAccuracy) + "%\n")
    print("This 'Faster Forward Selection' cost " + str(time.time() - start) + " seconds\n")
    if mode == 'faster':
        draw(drawAccuracy, drawFeatures, 'Faster Forward Selection')
    elif mode == 'normal':
        draw(drawAccuracy, drawFeatures, 'Forward Selection')

def backwardElimination(data, excludeFeatures):
    start = time.time()
    selectedFeatures = [i for i in range(1, len(data[0])) if i not in excludeFeatures]
    finalFeatures = selectedFeatures[:]
    finalAccuracy = 0
    drawAccuracy = []
    drawFeatures = [0]
    drawAccuracy.append(crossValidation(data, selectedFeatures))

    for i in range(1, len(data[0])):
        if len(selectedFeatures) == 1:
            drawFeatures.append(selectedFeatures[0])
            break

        bestAccuracy = 0
        pendingFeature = 0
        for j in range(1, len(data[0])):
            if j in selectedFeatures:
                selectedFeatures.remove(j)
                accuracy = crossValidation(data, selectedFeatures)
                print("        Using feature(s) " + str(selectedFeatures) + " accuracy is " + str(accuracy) + "%")
                selectedFeatures.append(j)
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    pendingFeature = j

        if pendingFeature != 0:
            selectedFeatures.remove(pendingFeature)
            drawFeatures.append(pendingFeature)
            drawAccuracy.append(bestAccuracy)

        if bestAccuracy >= finalAccuracy:
            finalAccuracy = bestAccuracy
            finalFeatures = selectedFeatures[:]
            print("\nFeature set " + str(selectedFeatures) + " was best, accuracy is " + str(bestAccuracy) + "%\n")
        else:
            print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            print("Feature set " + str(selectedFeatures) + " was best, accuracy is " + str(bestAccuracy) + "%\n")

    print("\nFinished search!! The best feature subset is " + str(finalFeatures) + ", which has an accuracy of " + str(finalAccuracy) + "%\n")
    print("This 'Faster Forward Selection' cost " + str(time.time() - start) + " seconds\n")
    defaultRate = crossValidation(data, [])
    if defaultRate < 50:
        drawAccuracy.append(100 - defaultRate)
    else:
        drawAccuracy.append(defaultRate)

    draw(drawAccuracy, drawFeatures, 'Backward Elimination')

def draw(accuracy, features, mode):
    if mode == 'Forward Selection' or mode == 'Faster Forward Selection':
        x_axis = [i for i in range(0, len(features))]
        plt.xlim(-0.6, len(features) - 0.4)
    elif mode == 'Backward Elimination':
        features.reverse()
        x_axis = [i for i in range (len(features) - 1, -1, -1)]
        plt.xlim(len(features) - 0.4, -0.6)

    for a, b in zip(x_axis, accuracy):
        plt.text(a, b + 0.1, '%d' % features[a], ha='center', va='bottom', fontsize=10)

    plt.ylim(0, 110)
    plt.bar(x_axis, accuracy, width=0.8, align='center', color='c')
    plt.xticks(x_axis, x_axis)
    global path
    plt.title(mode + ": " + str(path))
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy (%)")
    plt.show()

def main():
    global path
    path = input("Welcome to Bertie Woosters Feature Selection Algorithm.\nType in the name of the file to test : ")
    initMode = input("Type the number of the algorithm you want to run.\n\n1) Forward Selection\n2) Backward Elimination\n3) Bertie’s Special Algorithm.\n\n")

    with open(path, "r") as f:
        inputData = f.readlines()

    raw = [line.strip() for line in inputData]
    data = []
    for line in raw:
        line = line.split("  ")
        point = []
        for i in line:
            point.append(float(i))
        data.append(point)

    print("\nThis data set has", len(data[0]) - 1, "features (not including the class attribute), with", len(data),"instances.\n")
    acc = crossValidation(data, [i for i in range(1, len(data[0]))])
    print("Running nearest neighbor with all", len(data[0]) - 1, "features, using \"leaving-one-out\" evaluation, I get an accuracy of " + str(acc) + "%\n")
    print("Beginning search.\n")
    excludeFeatures = []
    if initMode == '1':
        forwardSelection(data, excludeFeatures)
    elif initMode == '2':
        backwardElimination(data, excludeFeatures)
    elif initMode == '3':
        forwardSelection(data, excludeFeatures, 'faster')

if __name__ == "__main__":
    main()