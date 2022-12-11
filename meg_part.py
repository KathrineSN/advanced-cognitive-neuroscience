# IMPORTS
import os
os.system('pip install mne')
import mne
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import permutation_test_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

def load_log_data():
    #csv_names = ["foodindex_0088_xsc.csv"]
    csv_names = ["foodindex_0085_DMS.csv", "foodindex_0086_H9T.csv", "foodindex_0087_QXD.csv","foodindex_0088_xsc.csv", "foodindex_0089_S3p.csv", "foodindex_0095_ZKY.csv"]
    food_index_list = []

    for i in range(len(csv_names)):

        file = pd.read_csv("/work/MEG Analysis - Kathrine/" + csv_names[i])

        food_index_list.append(list(file.event))

    return food_index_list

def preprocess_sensor_space_data(subject, date, raw_path,
                                 event_id=dict(random=7,
                                               food=29,
  ),
                                 h_freq=40,
                                 tmin=-0.200, tmax=0.500, baseline=(None, 0),
                                 reject=None, decim=1,
                                 return_epochs=False,
                                 food_index = []
                                 ):
    n_recordings = 6
    epochs_list = list()

    lower = 0
    upper = 60

    for recording_index in range(n_recordings): # ## loop from 0 to 5
        # decrease number depending on record
        food_index_temp = []

        #if subject == "0088" and recording_index == 2:
        #    pass

        #else:
        for i in food_index:
            print(i)
            if i >= lower and i <= upper:
                if recording_index >= 1:
                    food_index_temp.append(i- (60*(recording_index)))
                else:
                    food_index_temp.append(i)

        lower += 60
        upper += 60

        #true_indexes = np.array(food_index)-(60*(recording_index+1))
        fif_index = recording_index + 1 # files are not 0-indexed
        fif_fname = 'face_word_' + str(fif_index) 
        if subject == '0085': ## something went wrong with the first three rec.
            folder_name = '00' + str(fif_index + 3) + '.' + fif_fname
        else:
            folder_name = '00' + str(fif_index) + '.' + fif_fname
            
        full_path = join(raw_path, subject, date, 'MEG', folder_name,
                        'files', fif_fname + '.fif')
        raw = mne.io.read_raw(full_path, preload=True)
        raw.filter(l_freq=None, h_freq=h_freq, n_jobs=4)
        
        events = mne.find_events(raw, min_duration=0.002)
        # sorting so that we only get word events
        events = events[events[:,2]<14,:]
        
        for i in range(events.shape[0]):
            if i+1 in food_index_temp:
                events[i,2] = 29
            else:
                events[i,2] = 7

        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline,
                            preload=True, decim=decim)
        epochs.pick_types(meg=True)
        epochs.equalize_event_counts(event_ids=event_id, method='mintime')
        if return_epochs:
            epochs_list.append(epochs)
        else:
            if recording_index == 0:
                X = epochs.get_data()
                y = epochs.events[:, 2]
            else:
                X = np.concatenate((X, epochs.get_data()), axis=0)
                y = np.concatenate((y, epochs.events[:, 2]))

    
    if return_epochs:
        return epochs_list

    else:
        return X, y

def preprocess_source_space_data(subject, date, raw_path, subjects_dir,
                                 epochs_list=None,
                              event_id=dict(random=21,
                                            food=41,
 ),
                              h_freq=40,
                              tmin=-0.200, tmax=0.500, baseline=(None, 0),
                              reject=None, decim=1,
                              method='MNE', lambda2=1, pick_ori='normal',
                              label=None):
    if epochs_list is None:
        epochs_list = preprocess_sensor_space_data(subject, date, raw_path,
                                                   return_epochs=True)
    y = np.zeros(0)
    for epochs in epochs_list: # get y
        y = np.concatenate((y, epochs.events[:, 2]))
    
    if label is not None:
        label_path = join(subjects_dir, subject, 'label', label)
        label = mne.read_label(label_path)
    
    for epochs_index, epochs in enumerate(epochs_list): ## get X
        fwd_fname = 'face_word_' + str(epochs_index + 1) + '-oct-6-src-' + \
                    '5120-5120-5120-fwd.fif'
        fwd = mne.read_forward_solution(join(subjects_dir,
                                             subject, 'bem', fwd_fname))
        noise_cov = mne.compute_covariance(epochs, tmax=0.000)
        inv = mne.minimum_norm.make_inverse_operator(epochs.info,
                                                     fwd, noise_cov)
  
        
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2,
                                                     method, label,
                                                     pick_ori=pick_ori)    
        n_trials = len(stcs)
        for stc_index, stc in enumerate(stcs):
            this_data = stc.data
            if stc_index == 0:
                n_vertices, n_samples = this_data.shape
                this_X = np.zeros(shape=(n_trials, n_vertices, n_samples))        
            this_X[stc_index, :, :] = this_data
            
        if epochs_index == 0:
            X = this_X
        else:
            X = np.concatenate((X, this_X))
    return X, y

def collapse_events(y, new_value, old_values=list()):
    for old_value in old_values:
        y[y == old_value] = new_value
    return y

def simple_classication(X, y, participant, penalty='none', C=1.0, ):
    
    n_samples = X.shape[2]
    logr = LogisticRegression(penalty=penalty, C=C, solver='newton-cg')
    sc = StandardScaler() # especially necessary for sensor space as
                          ## magnetometers
                          # and gradiometers are on different scales 
                          ## (T and T/m)
    cv = StratifiedKFold()
    
    mean_scores = np.zeros(n_samples)
    
    for sample_index in range(n_samples):
        this_X = X[:, :, sample_index]
        sc.fit(this_X)
        this_X_std = sc.transform(this_X)
        scores = cross_val_score(logr, this_X_std, y, cv=cv, n_jobs=-1)
        mean_scores[sample_index] = np.mean(scores)
        
    '''    
    # added permutation test
    score, perm_scores, p_val = permutation_test_score(logr, X, y, cv=cv, n_permutations=100, n_jobs=-1)
    fig, ax = plt.subplots()
    ax.hist(perm_scores, bins=20, density=True)
    ax.axvline(score, ls="--", color="r")
    score_label = f"Score on original\ndata: {score:.2f}\n(p-value: {p_val:.3f})"
    ax.text(0.7, 10, score_label, fontsize=12)
    ax.set_xlabel("Accuracy score")
    _ = ax.set_ylabel("Probability")
    plt.savefig(f'/work/MEG Analysis - Kathrine/permutation_{participant}.png')
    '''

    return mean_scores

def plot_classfication(times, mean_scores, participant, title=None):
    plt.figure()
    plt.plot(times, mean_scores)
    plt.hlines(0.50, times[0], times[-1], linestyle='dashed', color='k')
    plt.ylabel('Proportion classified correctly')
    plt.xlabel('Time (s)')
    if title is None:
        plt.title('Random - Food')
    else:
        plt.title(title)
    
    if participant == "average":
        plt.title(f'Average classification of Food vs. Random of participant {participant}')
    

    plt.show()
    plt.savefig(f'/work/MEG Analysis - Kathrine/classification_{participant}.png')

def permutation_testing(X_lh_rh, collapsed_y, participant):
    # indledende øvelser fra simple class function (need to do it again, because it only functions locally in a function)
    X = X_lh_rh.copy()
    #print(X.shape)

    n_samples = X.shape[2]

    #standardizing the X
    sc = StandardScaler() # especially necessary for sensor space as
                            ## magnetometers
                            # and gradiometers are on different scales 
                            ## (T and T/m)
        
    for sample_index in range(n_samples):
        this_X = X[:, :, sample_index]
        sc.fit(this_X)
        this_X_std = sc.transform(this_X)

    # calculating the permutation test score
    score, permutation_scores, pvalue = permutation_test_score(
        LogisticRegression(penalty="l2", C=1e-3, solver='newton-cg'),
        this_X_std, collapsed_y, cv=StratifiedKFold(), n_permutations=1000, 
        n_jobs=1, random_state=0, verbose=0, scoring=None)
    print("Classification Accuracy: %s (pvalue : %s)" % (score, pvalue))

    #How many classes
    n_classes = np.unique(collapsed_y).size

    # round p val
    pvalue = round(pvalue,7)

    plt.hist(permutation_scores, 20, label='Permutation scores',
            edgecolor='black', color = 'xkcd:lightblue')
    ylim = plt.ylim()
    plt.plot(2 * [score], ylim, '--g', linewidth=3,
            label='Classification Score'
            ' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Chance level')

    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.title(f"Permutation test participant of participant {participant}")
    plt.suptitle("Classification of Food vs. Random in the insula")
    plt.savefig(f'/work/MEG Analysis - Kathrine/permutation_{participant}.png')

    plt.show()


if __name__ == '__main__':
    # Loading necessary log data
    food_index_list = load_log_data()

    # Preprocessing
    participants = ['0085','0086', '0087','0088', '0089', '0095']
    dates = ['20221004_000000','20221004_000000', '20221005_000000','20221005_000000', '20221007_000000', '20221007_000000']
    mean_scores_each_participant = []

    for i in range(len(participants)):
        print("___________________" + str(i) + "___________________")
        X_sensor, y = preprocess_sensor_space_data(participants[i], dates[i],
            raw_path='/work/220269/raw_MEG', decim=4, food_index=food_index_list[i]) 

        epochs_list = preprocess_sensor_space_data(participants[i], dates[i],
                raw_path='/work/220269/raw_MEG/',
                return_epochs=True, decim=4, food_index=food_index_list[i]) 

        # left hemisphere
        X_insula_lh, y = preprocess_source_space_data(participants[i],
                                                      dates[i],
        raw_path='/work/220269/raw_MEG/', 
        subjects_dir='/work/220269/freesurfer/',
        label='lh.insula.label', epochs_list=epochs_list)

        # right hemisphere
        X_insula_rh, y = preprocess_source_space_data(participants[i],
                                                      dates[i],
        raw_path='/work/220269/raw_MEG/',  
        subjects_dir='/work/220269/freesurfer/',
        label='rh.insula.label', epochs_list=epochs_list)

        # Collapsing events
        collapsed_y = collapse_events(y, 0, [7])
        collapsed_y = collapse_events(collapsed_y, 1, [29])

        X_both = np.concatenate((X_insula_lh,
                                            X_insula_rh), axis=1)

        mean_scores_sensor = simple_classication(X_both,
                                           collapsed_y, participants[i],
                                           penalty='l2', C=1e-3)
        # Performing classification
        '''
        mean_scores_sensor = simple_classication(X_sensor,
                                    collapsed_y, participants[i],
                                    penalty='l2', C=1e-3)
        '''
        mean_scores_each_participant.append(mean_scores_sensor)
        
        # Plotting classification
        plot_classfication(epochs_list[0].times, mean_scores_sensor, participants[i])

        permutation_testing(X_both, collapsed_y, participants[i])
    
    mean_scores_avg = np.mean(np.array(mean_scores_each_participant), axis = 0) 

    plot_classfication(epochs_list[0].times, mean_scores_avg, participant = "average")