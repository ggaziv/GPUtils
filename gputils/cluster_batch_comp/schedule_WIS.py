#!/usr/bin/env python
"""
    Batch task computation over cluster.
    Date created: 17/4/18
    Python Version: ?
"""

__author__ = "?"
__credits__ = ["?"]
__email__ = "?"

import argparse
import json
import os
import datetime
import subprocess



def get_run_id(expIdentifier):
    filename = os.path.expanduser("~/logs/" + expIdentifier + ".expts")
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts) / 4
        return run_id

        
        
        
parser = argparse.ArgumentParser()
parser.add_argument("-params", "--params", default="{}", type=str, help="JSON string of arguments",nargs='?')
parser.add_argument("-schedules", "--schedules", default=1, type=int, help="Number of jobs you wish to schedule in a sequence",nargs='?')
parser.add_argument("-expIdentifier", "--expIdentifier", default='', type=str, help="Experimet identifier name",nargs=1)
parser.add_argument("-expFileName", "--expFileName", type=str, help="File name to call",nargs=1)
parser.add_argument("-cpu", action='store_true', help="Use a cpu quesue instead of gpu")
args = parser.parse_args()
print(args.params)

params = json.loads(args.params)
expFileName = args.expFileName[0]
expIdentifier = args.expIdentifier[0]

# print(expFileName)
if expFileName==None:
	expFileName = expIdentifier
elif expFileName[-3:]=='.py':
    expFileName = expFileName[:-3]
run_id = get_run_id(expIdentifier)
print("Scheduling %s_%d" % (expIdentifier, run_id))
# Copying code to a directory to prevent clashes with future jobs
command = "cp -r Code ~/codes/" + expIdentifier + "_" + str(run_id)
print(subprocess.check_output(command, shell=True))

params_2_change_in_subsequent = {}
# Write the scheduler scripts
for i in range(args.schedules):
    script = "# !/bin/tcsh\n\n"
    script += "# $ -S /bin/bash\n"
    script += "setenv TERM xterm\n"
    script += "hostname\n"
    script += "source cluster.sh\n"
    script += \
		"python \\\n" + \
        os.path.expanduser("~/codes/" + expIdentifier + "_" + str(run_id) + "/" +expFileName+".py")+ " \\\n"
    for k, v in params.items():
        if i>0 and k in params_2_change_in_subsequent.keys():
            v = params_2_change_in_subsequent[k]
            # print('Changed value of %s to %s'%(str(k),str(v)))
        if type(v) == bool and v is True:
            script += "-" + k + " \\\n"
        elif type(v) == bool and v is False:
            pass
        elif type(v) == dict:
            script += "-" + k + " '" + json.dumps(v) + "' \\\n"
        else:
            script += "-" + k + " " + str(v) + " \\\n"
    # script += "-run_id " + str(run_id) + "\n"

    # Write schedule script
    with open(os.path.expanduser('~/schedulers/' + expIdentifier + "_" + str(run_id) + "_" + str(i) + ".sh"), 'w') as f:
        f.write(script)

# Update experiment logs
filename = "~/logs/" + expIdentifier + ".expts"
output = \
    str(run_id) + "\n" + \
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" + \
    "python schedule.py -params '" + str(args.params) + \
    "' -schedules " + str(args.schedules) + "\n\n"
# print(output, filename)
with open(os.path.expanduser(filename), "a") as f:
    f.write(output)

# schedule jobs
# if args.schedules == 1:
output_folders = 'Jobs'
command = \
    "qsub -N %s -q %s.q -o %s -e %s "%(expIdentifier,'all' if args.cpu else 'gpu',output_folders,output_folders) + "~/schedulers/" + \
    expIdentifier + "_" + str(run_id) + "_0" + ".sh"
print("Scheduling job")
print(subprocess.check_output(command, shell=True))
# else:
#     for i in range(args.schedules):
#         command = \
#             "sbatch -J " + expIdentifier + "_" + str(run_id) + " -d singleton " + os.path.expanduser("~/schedulers/" + \
#             expIdentifier + "_" + str(run_id) + "_" + str(i) + ".sh")
#         print("Scheduling job #%d" % i)
#         print(subprocess.check_output(command, shell=True))
