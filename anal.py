import os,sys
import operator
import cPickle as cp
import re
from collections import Counter
get_results = False
if get_results:
	d = os.path.join(os.getcwd(),"results")
	results = {}
	for k in [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]:
		try:
			# print k
			path = os.path.join(d,k)
			files = os.listdir(path)
			results_file = cp.load(open(os.path.join(path,"results.pickle")))
			# print results_file
			results[k]={}
			results[k]["mean_error"]=results_file["mean_error"]
			m = re.match("nh-([0-9]+)_nr-([0-9]+)_nc-([0-9]+)_l2-([0-9\.]+)_lr-([0-9\.]+)_lrpt-([0-9\.]+)_trial-([0-9\.]+)",k)
			results[k]['nh'] = m.group(1)
			results[k]["nr"] = m.group(2)
			results[k]["nc"] = m.group(3)
			results[k]["l2"] = m.group(4)
			results[k]["lr"] = m.group(5)
			results[k]["lrpt"]= m.group(6)
			results[k]["trial"] = m.group(7)
			for var in ["nh","nr","nc","l2","lr","lrpt","trial"]:
				results[k][var] = float(results[k][var])
				# print results
			# print files
		except:
			try:
				del results[k]
			except:
				pass
else:
	results = cp.load(open("apocrita_results.pickle"))

sorted_results_keys = sorted(results.iteritems(), key=operator.itemgetter(1))
sorted_results_keys = sorted(sorted_results_keys, key=lambda k:k[1]["mean_error"])

for i in sorted_results_keys:
	print i[0],":",i[1]["mean_error"]

for var in ["nh","nr","nc","l2","lr","lrpt","trial"]: 
	t_var = [results[key][var] for key in results.keys()]
	print "var:",var
	print Counter(t_var)