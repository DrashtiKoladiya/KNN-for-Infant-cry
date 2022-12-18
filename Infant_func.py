import numpy as np
import librosa
import pickle
import os

def predict_test(sound_file):
    audio,sr=librosa.load(sound_file)
    mfccs=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=13)
    mfccs_processed=np.mean(mfccs.T,axis=0)
    test_extracted = mfccs_processed
    test_extracted=test_extracted.reshape((1,-1))
    classifier_mod = pickle.load(open('knnpickle_file','rb'))
    y_pred=classifier_mod.predict(test_extracted)
    if y_pred==0:
        label='Normal cry'
    elif y_pred==1:
        label='Pathology cry'
    return label    


def get_file_paths(training_path):
        # get file paths
        files   = [ os.path.join(training_path, f) for f in os.listdir(training_path) ]
        return files

normal_training_path = r"C:\Drashti\Website\Infant\baby_test_data\normal"
pathological_training_path = r"C:\Drashti\Website\Infant\baby_test_data\pathlogy"

# Finding Accuracy of Normal Cry
total_sample = 0
error = 0
files = get_file_paths(normal_training_path)
for file in files:
    total_sample += 1
    
    identified_class = predict_test(file)
    expected_class = 'Normal cry'

    if identified_class != expected_class: error += 1

accuracy_n     = ( float(total_sample - error) / float(total_sample) ) * 100
accuracy_normal = "*** Accuracy for class Normal = " + str(round(accuracy_n, 3)) + "% ***"


# Finding Accuracy of Pathlogical Cry
total_sample = 0
error = 0
files = get_file_paths(pathological_training_path)
for file in files:
    total_sample += 1
    
    identified_class = predict_test(file)
    expected_class = 'Pathology cry'

    if identified_class != expected_class: error += 1

accuracy_p     = ( float(total_sample - error) / float(total_sample) ) * 100
accuracy_pathology  = "*** Accuracy for class Pathology  = " + str(round(accuracy_p, 3)) + "% ***"

total_accuracy = (accuracy_n + accuracy_p)/2
accuracy_t = "*** Total Accuracy = " + str(round(total_accuracy, 3)) + "% ***"
print(accuracy_normal)
print(accuracy_pathology)
print(accuracy_t)