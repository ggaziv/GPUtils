#!/usr/bin/env python
"""
    Batch task computation over cluster.
    Date created: 17/4/18
    Python Version: ?
"""

__author__ = "?"
__credits__ = ["?"]
__email__ = "?"

import os
# import numpy as np
import subprocess
import sys


def at2doubleQuates(input):
    return input.replace('@','\"')

what_2_call = 'stl10_Amit_grads'#'cifar100_Amit_grads'#'test'#''stl10_Amit_grads'#amit_stats
only_print = True
job_counter = 0

if what_2_call=='stl10_Amit_grads':
    SVHN_exp = False
    cifar100_exp = False
    MC_Dropout_exp = False
    for manip in ['0']:#'0','29','3','52','9']:
        for set in ['detectorTrain']:#,'train'
            for model in ['None']:#d,'dist','AT']:#,
                for amitFeatures in [False]:
                    command = "python schedule_WIS.py -expIdentifier stl10_grads -expFileName elu_network_stl10_at"+\
                        " -params '{@set@:@%s@,@amitModel@:@%s@"%(set,model)+\
                        (",@gradFeatures@:@%s@"%(manip) if manip is not None else "")+\
                        (",@amitFeatures@:@@" if amitFeatures else "")+(",@SVHN@:@@" if SVHN_exp else "")+\
                        (",@cifar100@:@@" if cifar100_exp else "")+(",@MCD@:@@" if MC_Dropout_exp else "")+"}'"
                    job_counter+=1
                    if only_print:
                        print(at2doubleQuates(command))
                    else:
                        print(subprocess.call(at2doubleQuates(command), shell=True))
elif what_2_call == 'amit_stats':
    SVHN_exp = False
    cifar100_exp = True
    for MC_Dropout_exp in [False,True]:
        for data_set in ['STL10_Amit']:#''STL10_Amit']:#,'cifar100_Amit']:
            for set in ['train', 'detectorTrain']:
                for amitModel in ['AT','dist','None']:#,'None']:
                    for errorIs in ['novel','normal','ignored']:
                        if errorIs!='novel' and not (SVHN_exp or cifar100_exp):
                            continue
                        command = "python schedule_WIS.py -expIdentifier AmitStats -expFileName stl-novelty" + \
                                  " -params '{@data_set@:@%s@,@set@:@%s@,@amitModel@:@%s@,@errorIs@:@%s@" % (
                                  data_set, set, amitModel,errorIs) +(",@SVHN@:@@" if SVHN_exp else "")+\
                                  (",@cifar100@:@@" if cifar100_exp else "")+(",@MCD@:@@" if MC_Dropout_exp else "")+"}'"+" -cpu"
                        job_counter += 1
                        if only_print:
                            print(at2doubleQuates(command))
                        else:
                            print(subprocess.call(at2doubleQuates(command), shell=True))
elif what_2_call == 'cifar100_Amit_grads':
    MC_Dropout_exp = False
    for manip in [None]:
        for set in ['detectorTrain']:
            for model in ['AT','None','dist']:
                for amitFeatures in [False]:
                    command = "python schedule_WIS.py -expIdentifier cifar100_grads -expFileName elu_network_cifar" + \
                              " -params '{%s@set@:@%s@,@amitModel@:@%s@" % (
                              "@gradFeatures@:@%s@,"%(manip) if manip is not None else "", set, model) + \
                              (",@amitFeatures@:@@" if amitFeatures else "")+(",@MCD@:@@" if MC_Dropout_exp else "")+ "}'"
                    job_counter += 1
                    if only_print:
                        print(at2doubleQuates(command))
                    else:
                        print(subprocess.call(at2doubleQuates(command), shell=True))
elif what_2_call=='test':
    for num in range(4):
        command = "python schedule_WIS.py -expIdentifier hello_world -expFileName hello_world"+\
            " -params '{@num@:%d}'"%(num)
        job_counter+=1
        if only_print:
            print(at2doubleQuates(command))
        else:
            print(subprocess.call(at2doubleQuates(command), shell=True))
elif what_2_call=='classifier_eval':
    onlyLabels = list(range(10))
    for onlyLabel in onlyLabels:
        command = "python schedule_WIS.py -expIdentifier STL10_eval_"+str(onlyLabel)+" -expFileName stl10_classifier"+\
        " -params '{@onlyLabel@:%d"%(onlyLabel)+",@state@:@eval@}'"
        job_counter+=1
        if only_print:
            print(at2doubleQuates(command))
        else:
            print(subprocess.check_output(at2doubleQuates(command), shell=True))
elif what_2_call=='evalGrad':
    # modificationNums = range(0,3)
    modifications = ['0_29_3']#'0_3','0','3','0_29']
    # modificationNums = range(3,5)
    sets = ['val','train','detectorTrain']
    # for modificationNum in modificationNums:
    for modification in modifications:
        for cur_set in sets:
            for cur_augment in [True,False]:
                if cur_set=='val' and cur_augment:
                    continue
                if cur_set!='val' and not cur_augment:
                    continue
                # featureType = 'augmentedLogits'+''.join([str(digit) for digit in range(0,modificationNum+1)])
                featureType = 'augmentedLogits'+modification
                # featureType = 'augmentedLogits0'+str(modificationNum)
                # " -params '{@state@:@evalGrad@,@modifiedSet@:@withAugm@,@set@:@"+cur_set+"@,@gradFeatures@:@"+featureType+"@"+(",@augment@:45" if cur_augment else "")+"}' -schedules 2"
                command = "python schedule_WIS.py -expIdentifier STL10_"+featureType+" -expFileName stl10_classifier"+\
                    " -params '{@state@:@evalGrad@,@set@:@"+cur_set+"@,@gradFeatures@:@"+featureType+"@"+(",@augment@:45" if cur_augment else "")+"}' -schedules 2"
                job_counter+=1
                if only_print:
                    print(at2doubleQuates(command))
                else:
                    print(subprocess.check_output(at2doubleQuates(command), shell=True))
elif what_2_call=='train_stl10_classifier':
    state = 'evalGrad'#'train'#'evalGrad'
    globally_excluded_labels = [0,1]
    # if state=='evalGrad':
    all_labels = sorted([i for i in range(10) if i not in globally_excluded_labels])
    pairs = [[[j,i] for j in range(all_labels[0],i) if j not in globally_excluded_labels] for i in all_labels]
    labels2exclude = []
    for pairs_list in pairs:
        for pair in pairs_list:
            labels2exclude.append(pair)
    # else:
    #   labels2exclude
    for pair in labels2exclude:
        cur_labels_str = ' '.join([str(i) for i in globally_excluded_labels+pair])
        command = "python schedule_WIS.py -expIdentifier %s_STL10class -expFileName stl10_classifier -params '{@state@:@%s@,@cifarModel@:@@,@exLabels@:@%s@%s}'"%(''.join([str(i) for i in pair]),state,cur_labels_str,
            (",@exEvalGradLabels@:@%s@,@set@:@val4novelty@,@augment@:8,@manipulations@:@0 29 3 52 9@"%(' '.join([str(i) for i in globally_excluded_labels])) if state=='evalGrad' else ''))
        job_counter+=1
        if only_print:
            print(at2doubleQuates(command))
        else:
            print(subprocess.check_output(at2doubleQuates(command), shell=True))
elif what_2_call=='train_CIFAR10':
    data_sets = os.listdir(os.path.expanduser('/home-nfs/ybahat/experiments/GMMonVgg16/CIFAR10_alex/modified_CIFAR'))
    data_sets = [cur_set for cur_set in data_sets if ('Exc' in cur_set or 'Original' in cur_set)]
    print(data_sets)
    dirs2discard = ['Original','Original1','Original0','Exc_Frog','Exc_Automobile']
    for cur_set in data_sets:
        if cur_set in dirs2discard:
            continue
        command = "python schedule_WIS.py -expIdentifier "+cur_set+"_NetTrain -expFileName cifar10_train -params '{@dataDir@:@"+\
        cur_set+"@}' -schedules 8"
        job_counter+=1
        if only_print:
            print(at2doubleQuates(command))
        else:
            print(subprocess.check_output(at2doubleQuates(command), shell=True))
elif what_2_call=='results':
    GMM_name_filter = ['Exc_Automobile']
    GMM_names = [file[:-4] for file in os.listdir(os.path.expanduser('/home-nfs/ybahat/experiments/GMMonVgg16/CIFAR10_alex/GMMs')) if file[-4:]=='.npz']
    GMM_names = [file for file in GMM_names if all([name_filter in file for name_filter in GMM_name_filter])]
    for GMM_name in GMM_names:
        for cur_set in ['train','val']:
            command = "python schedule_WIS.py -expIdentifier GMMeval -expFileName imagesHist_CIFAR10 -params '{@GMModel@:@"+\
            GMM_name+"@,@numPatches@:10000,@set@:@"+cur_set+"@}'"
            job_counter+=1
            if only_print:
                print(at2doubleQuates(command))
            else:
                print(subprocess.check_output(at2doubleQuates(command), shell=True))
elif what_2_call=='Grad_Classifer_Train':
    initLR = [0.005]
    BSZ = [32]
    Lwidths = [[5.32,5.64,60,60]]
    onlyLabels = list(range(10))
    train = True
    for initLR_ in initLR:
        for BSZ_ in BSZ:
            for Lwidths_ in Lwidths:
                for onlyLabel in onlyLabels:
                    # " -expFileName NeuralNet_classifier -params '{@NNDir@:@Original_discard50@,@augment@:8,@imagesDir@:@Original@,"+("@train@:@@," if train else '')+
                    command = "python schedule_WIS.py -expIdentifier "+str(BSZ_)+"_%.0e_LinGradClass"%(initLR_)\
                    +" -expFileName NeuralNet_classifier -params '{@NNDir@:@STL10_Original@,@augment@:10,@imagesDir@:@STL10_Original@,"\
                    +("@train@:@@," if train else '')+"@BSZ@:@%d"%(BSZ_)+"@,@LRinit@:@"+"%f@,@Lwidths@:@%s@,@normalization@:@batchInput@,@DO@:@@,@CB@:@@,@onlyLabel@:%d}'"\
                        %(initLR_,(' ').join([str(num) for num in Lwidths_]),onlyLabel)
                    job_counter+=1
                    if only_print:
                        print(at2doubleQuates(command))
                    else:
                        print(subprocess.check_output(at2doubleQuates(command), shell=True))

elif what_2_call=='per_label_results':
    onlyLabels = list(range(10))
    Normalities = [1,0]
    for Normality in Normalities:
        for onlyLabel in onlyLabels:
            command = "python schedule_WIS.py -expIdentifier "+'col_res_l'+str(onlyLabel)+"_norm%d"%(Normality)+\
            " -expFileName collect_results -params '{@onlyLabel@:%d,@Normality@:%d}'"%(onlyLabel,Normality)
            job_counter+=1
            if only_print:
                print(at2doubleQuates(command))
            else:
                print(subprocess.check_output(at2doubleQuates(command), shell=True))
if only_print:
    print('Printed %d jobs.' % (job_counter))
else:
    print('Submitted %d jobs.'%(job_counter))
