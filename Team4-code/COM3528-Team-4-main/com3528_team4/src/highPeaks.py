from scipy.signal import find_peaks

def find_high_peaks(cross_correlation):
		# Find peaks in the cross-correlation result
		peaks, _ = find_peaks(cross_correlation, height=0.5)

		return peaks