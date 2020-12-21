import pandas as pd
import numpy as np
import csv

def base_dataset(df, scans):
	cn, ad = [], []
	keys = ["PTID", "IMAGEUID", "LDELTOTAL", "MMSE", "CDRSB", "mPACCdigit", "mPACCtrailsB", "DX"]

	with open('data.csv', mode = 'w') as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=keys)
		writer.writeheader()

		for scan in scans:
			frame = df[df["PTID"] == scan]
			bl = frame[frame["VISCODE"] == "bl"]

			if len(bl["IMAGEUID"].values) > 0:
				data = {}
				for key in keys:
					data[key] = bl[key].values[0]

				writer.writerow(data)




def main():
	keys = ["VISCODE", "PTID", "IMAGEUID", "LDELTOTAL", "MMSE", "CDRSB", "mPACCdigit", "mPACCtrailsB", "DX"]
	df = pd.read_csv('ADNIMERGE.csv', low_memory = False).dropna(subset = keys) 

	cn_mci, mci_ad, cn_cn, mci_mci, ad_ad = set(), set(), set(), set(), set()
	for scan in df.iloc:
		if scan["DX_bl"] == "CN" and scan["DX"] == "MCI":
			cn_mci.add(scan["PTID"])
		elif (scan["DX_bl"] == "LMCI" or scan["DX_bl"] == "EMCI") and scan["DX"] == "Dementia":
			mci_ad.add(scan["PTID"])

		d = np.unique(df[df["PTID"] == scan["PTID"]]["DX"].values)

		if len(d) == 1 and d[0] == "CN":
			cn_cn.add(scan["PTID"])
		elif len(d) == 1 and d[0] == "MCI":
			mci_mci.add(scan["PTID"])
		elif len(d) == 1 and d[0] == "Dementia":
			ad_ad.add(scan["PTID"])

	base_dataset(df, cn_cn.union(ad_ad))

	'''
	print(len(cn_mci))	# 96
	print(len(cn_cn))	# 695
	print(len(ad_ad))	# 396

	print(len(mci_ad))	# 339
	print(len(mci_mci))	# 635
	
	accept = nonconverters.union(converters)
	rejects = set()

	for ptid in df["PTID"].unique():
		if ptid not in accept:
			rejects.add(ptid)

	print(len(rejects))

	for conv in rejects:
		c = df[df["PTID"] == conv]
		
		data = [[row[key] for key in keys] for row in c.iloc]
		data = sorted(data, key= lambda x: 0 if x[0] == "bl" else int(x[0][1:]))

		for d in data:
			print(d)
		
		input()
	'''


if __name__ == '__main__':
	main()