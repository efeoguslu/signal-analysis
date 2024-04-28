import matplotlib.pyplot as plt
import numpy as np
import math
import accelLogParser
import os
import sys
from typing import List

default_target_folder_path = "C:\\Users\\efeog\\OneDrive\\Masaüstü\\signal_analysis"

fig, ax1 = plt.subplots()
ax1.set_title('Raw Data, Processed Data, Normalized Angles and Speed')
ax1.set_xlabel('Seconds (s)')
show_cpp = False

def process_signal(signal, _coef=0.1):
    acc_val=signal[0]
    coef = _coef
    outdata=[]
    for x in signal:
        acc_val = (acc_val * (coef)) + ( x * (1-coef))
        outdata.append(acc_val)
    return outdata


def prepare_data(filePath):
    file = open(filePath)
    all_data = file.readlines()
    out_data = []
    for line in all_data:        
        out_data.append(float((line.split(","))[1].strip()))
        
    return out_data

def prepare_data_from_logs(target) -> List[dict]:
    logsDictList = accelLogParser.parseLinesFromLog(os.path.join(target,"allSensorLogFile.txt"))
    return logsDictList

def prepare_data_from_bump_logs(target) -> List[dict]:
    logsDictList = accelLogParser.parseLinesFromLog(os.path.join(target,"bumpCountLogFile.txt"))
    return logsDictList

def active_process(signal, threshold, window_size = 128, overlap = 64):
    signal_size = len(signal)
    iterations = int((signal_size-window_size)/(window_size-overlap))
    out_data = []

    signal = np.array(signal)
    signal = signal - 1 

    print(signal_size)
    for i in range (iterations):
        start_sample = (window_size-overlap)*i
        end_sample = start_sample+window_size
        window_data = signal[ start_sample : end_sample]
        
        coef = 1.0
        # if(max(window_data) > (threshold)) or ( abs(min(window_data)) > (threshold)):
        if( (max(window_data)-min(window_data)) > threshold ):
            coef = 1.2 ## small gain signal
        else:
            coef = 0.4
            ## attenue signal
        
        for j in range(start_sample,end_sample):
            signal[j] = signal[j]*coef
    return signal


def extract_values(dataList, *keys):
    lists_of_values = {}
    
    # l_name = [d.get('Name') for d in l]
    # print(l_name)

    for dictionary in dataList:

        for key in keys:

            if key in dictionary:
                if key not in lists_of_values:
                    lists_of_values[key] = [dictionary[key]]
                else:
                    lists_of_values[key].append(dictionary[key])
    
    return lists_of_values

def extract_data_by_keys(list_of_dicts, keys):
    # Initialize an empty list to hold the result
    result = []
    
    # Iterate over each key in the keys list
    for key in keys:
        # Use list comprehension to extract values for the current key from all dictionaries
        values = [float(d[key]) for d in list_of_dicts if key in d]
        # Append the list of values to the result
        result.append(values)
    
    return result


def plot_list_of_lists(list_of_lists, time, labels=None):

    # Iterate over the list of lists
    for i, sublist in enumerate(list_of_lists):
        # Plot each sublist and specify label if available
        if labels:
            ax1.plot(time, sublist, label=labels[i])
        else:
            ax1.plot(time, sublist)
    
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    # ax1.set_title('Raw Data, Processed Data, Normalized Angles and Speed')
    ax1.grid(True)

    if labels:
        ax1.legend(loc='upper right', fontsize='small', shadow=True, handlelength=1.5)
        # plt.legend()


def get_nmea_file(target):
    # list files
    file_names = os.listdir(target)

    # iterate over the file names
    for file_name in file_names:
        # check if the file name ends with ".nmea"
        if file_name.endswith(".nmea"):
            return file_name



def update_target_folder_path():

    arguments = sys.argv

    # Check if there are 2 or more command-line arguments
    if len(arguments) > 2:
        print("Warning: Please put your file path in \" \" and try again.")
        sys.exit()

    # Check if command-line arguments were provided
    elif len(arguments) == 2:
        # If an argument is provided, use it as the target folder path
        if not os.path.isdir(sys.argv[1]):
            print("Error: The specified folder path is not valid.")
            sys.exit()

        return sys.argv[1]
    else:
        # Otherwise, use the default target folder path
        return default_target_folder_path

# ----------------------------------------------------------------------------------------------------------------------



def plot_processed_signals(raw_signal, legend_name, iir_coef, isactive, threshold=None, window_size=None, overlap=None):
    
    # step_sec = (1.0/75.0)    
    # time = np.arange(start = 0, step = step_sec, stop = step_sec*len(raw_signal))
    
    time = get_time(raw_signal)

    # Plot each sublist and specify label if available
    iir_filtered_signal = process_signal(raw_signal, iir_coef) 
    ax1.plot(time, iir_filtered_signal, label = (legend_name + "_iir_filtered"))
    if isactive:

        if threshold is not None and window_size is not None and overlap is not None:
            active_signal = active_process(iir_filtered_signal, threshold, window_size, overlap)
            plt.plot(time, active_signal, label = (legend_name + "_active_filtered"))

    ax1.legend(loc='upper right', fontsize='small', shadow=True, handlelength=1.5)


# ----------------------------------------------------------------------------------------------------------------------

def plot_speed(dataList, gpsList, time):

    speed_data = []

    for i in dataList:
        gpsEnt = accelLogParser.getGpsEntryAtTime(gpsList, i["time"])
        speed_value = 0
        
        if (gpsEnt != None):
           speed_value = gpsEnt["speed"]
        
        speed_data.append(float(speed_value))

    speed_color = 'tab:red' 
    ax2 = ax1.twinx()
    ax2.set_ylabel('Speed (km/h)', color=speed_color)
    ax2.tick_params(axis='y', labelcolor=speed_color)
    ax2.plot(time, speed_data, color=speed_color, label = "Speed")
    ax2.legend(loc = 'upper left')

# ----------------------------------------------------------------------------------------------------------------------

def plot_normalized_angles(angles, time, labels=None):
    # Iterate over the list of lists
    for i, sublist in enumerate(angles):
        # Plot each sublist and specify label if available
        if labels:
            ax1.plot(time, sublist, label=labels[i])
        else:
            ax1.plot(time, sublist)
    
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Angles')
    # ax1.set_title('Plot of Angles')
    ax1.grid(True)

    if labels:
        ax1.legend(loc='upper right', fontsize='small', shadow=True, handlelength=1.5)

# ----------------------------------------------------------------------------------------------------------------------

def plot_comp_vector(comp_vector, time, labels=None):
    
    if labels:
        ax1.plot(time, comp_vector, label=labels)
    else:
        ax1.plot(time, comp_vector)
    
    # ax1.set_xlabel('X-axis')
    # ax1.set_ylabel('Angles')
    # ax1.set_title('Plot of Angles')
    ax1.grid(True)

    if labels:
        ax1.legend(loc='upper right', fontsize='small', shadow=True, handlelength=1.5)

# ----------------------------------------------------------------------------------------------------------------------

def get_time(data):
    step_sec = (1.0 / 75.0)
    return np.arange(start = 0, step = step_sec, stop = step_sec*len(data))

# ----------------------------------------------------------------------------------------------------------------------

def normalize_list(lst):

    normalized_lst = []
    for sub_lst in lst:
        min_val = min(sub_lst)
        max_val = max(sub_lst)
        normalized_sub_lst = [(x - min_val) / (max_val - min_val) for x in sub_lst]
        normalized_lst.append(normalized_sub_lst)

    return normalized_lst

# ----------------------------------------------------------------------------------------------------------------------


def create_compound_vector(list_of_lists):

    vector_size = len(list_of_lists[0])
    
    compound_vector = []
    
    for index in range(vector_size):
        sum_of_squares = sum([vector[index] ** 2 for vector in list_of_lists])
        compound_vector_element = math.sqrt(sum_of_squares)
        compound_vector.append(compound_vector_element)
    
    return compound_vector


# ----------------------------------------------------------------------------------------------------------------------

def final_plot(raw_signal, legend_name, isactive, threshold=None, window_size=None, overlap=None):
    
    # step_sec = (1.0/75.0)    
    # time = np.arange(start = 0, step = step_sec, stop = step_sec*len(raw_signal))
    
    time = get_time(raw_signal)

    # Plot each sublist and specify label if available
    # iir_filtered_signal = process_signal(raw_signal, iir_coef) 

    ax1.plot(time, raw_signal, label = legend_name)
    if isactive:
        if threshold is not None and window_size is not None and overlap is not None:
            active_signal = active_process(raw_signal, threshold, window_size, overlap)
            plt.plot(time, active_signal, label = (legend_name + "_active_filtered"))

    ax1.legend(loc='upper right', fontsize='small', shadow=True, handlelength=1.5)


def get_bump_seconds(input_list):
    return [x / 75 for x in input_list]

def main():
    
    target_folder_path = update_target_folder_path()
  
    gpsList = accelLogParser.parseGpsLinesFromLog(os.path.join(target_folder_path, get_nmea_file(target_folder_path)))
    

    # print(gpsList)
    
    

    dataList = prepare_data_from_logs(target_folder_path)
    bumpList = prepare_data_from_bump_logs(target_folder_path)

    keys = ['comp_vector']

    extracted_data = extract_data_by_keys(dataList, keys)

    angles = extract_data_by_keys(dataList, ['roll_angle', 'pitch_angle'])

    normalized_angles = normalize_list(angles)

    time = get_time(extracted_data[0])

    comp_vec = create_compound_vector(extract_data_by_keys(dataList, ['ax_rotated', 'ay_rotated', 'az_rotated']))

    # plot_list_of_lists(extracted_data, time, keys)                                         # ham data plotları

    signal_to_be_plotted = extracted_data[0]

    # plot_processed_signals(signal_to_be_plotted, str(keys[0]), 0.9, True, 0.15, 50, 30)    # işlenmiş data plotları


    buton_data = extract_data_by_keys(dataList, ['button_state'])[0]
    bump_data = extract_data_by_keys(bumpList, ['sample'])[0]

    bump_data = get_bump_seconds(bump_data)
    
    buton_set = False
    for i in range(len(signal_to_be_plotted)):

        if(buton_data[i] == 1) and (not buton_set):            
            plt.axvline(time[i], color='black')
            buton_set = True
        if(buton_data[i] == 0) and buton_set:
            buton_set = False

    for x in bump_data:
        plt.axvline(x=x, color='r', linestyle='--', linewidth=1.0)  # Plot vertical lines

    # plot_speed(dataList, gpsList, time)
    # plot_normalized_angles(normalized_angles, time, ['roll_angle', 'pitch_angle'])
    # plot_comp_vector(comp_vec, time, 'Compound Vector')
    # plot_processed_signals(extracted_data[0], 'comp_vec', 0.9, True, 0.15, 50, 30)    # işlenmiş data plotları

    final_plot(signal_to_be_plotted, "recorded_compvec", True, 0.5, 50, 35)
    plt.show()
    exit()


    

    # get_dict_keys(dataList)

    keys = ['elma', 'armut', 'kivi']
    
    extracted_data = extract_data_by_keys(dataList, keys)

    plot_list_of_lists(extracted_data)

    exit()
    step_sec = (1.0/75.0)
    time = np.arange(start=0, step=step_sec, stop= step_sec*len(extracted_data[0]))

    # print(extracted_data)

    plot_list_of_lists(extracted_data, time)
    
    
    exit()

    result = extract_values(dataList, "az")

    print(str(result))

    # print(str(dataList))

    exit()
    # call the extract_values function with dataList and keys
    result = extract_values(dataList, "az")

    # print the result
    print(result)

    prep_data = []
    buton_data = []    
    speed_data = []

    prep_data_legend = "az_rotated"
    button_state_legend = "button_state"

    prep_data_key = prep_data_legend
    button_state_key = button_state_legend

    for i in dataList:
        prep_data.append(float(i[prep_data_key]))
        buton_data.append(int(i[button_state_key]))
        
        gpsEnt = accelLogParser.getGpsEntryAtTime(gpsList,i["time"])
        speed_value = 0
        
        if (gpsEnt != None):
           speed_value = gpsEnt["speed"]
        
        speed_data.append(float(speed_value))
        # print("time:"+i["time"]+ "speed:"+str(speed_value))

        
    
    
    # HAM DATA ARRAY VER 
    # alacağımız parametreler: legend için isim, HAM DATA ARRAY, iir coef, active_process_enable(boolean), active threshold, active window_size, active overlap 
    # SADECE plot fonk'u şimdilik 

    step_sec = (1.0/75.0)

    coef_09 = process_signal(prep_data,0.9)                                       # IIR 
    
    time = np.arange(start=0, step=step_sec, stop= step_sec*len(prep_data) )
    
    active_signal = active_process(coef_09, 0.15, window_size=50, overlap=30)     # ACTIVE PROCESSING
    


    procfile=open("procData.txt","wt")
    for i in active_signal:
        procfile.write(str(i)+"\n")
    procfile.close()
    
    
    prepDataFile=open("prepData.txt","wt")
    for i in prep_data:
        prepDataFile.write(str(i)+"\n")
    prepDataFile.close()
    
    
    if show_cpp:    
        cppProcFile = open("processedDataCpp.txt","rt")
        cppLines=cppProcFile.readlines()
        cppProcFile.close()
        cppProcValues = []
        for line in cppLines:
            cppProcValues.append(float(line.strip()))
            
        while(len(cppProcValues) < time.size):
            cppProcValues.append(0)

        cppiirFile = open("iirDataCpp.txt","rt")
        cppiirLines=cppiirFile.readlines()
        cppiirFile.close()
        cppiirValues = []
        for line in cppiirLines:
            cppiirValues.append(float(line.strip()))
            
        while(len(cppiirValues) < time.size):
            cppiirValues.append(0)
        print("cppProcValues size:",len(cppProcValues))

    print("active signal size:",active_signal.size)
    print("time size:",time.size)
    
    # print("cppProcValues:",cppProcValues)
    fig, ax1 = plt.subplots()
    
    # plt.figure()
    ax1.plot(time, prep_data, label = prep_data_legend)    
    ax1.plot(time, coef_09, label = "IIR Filter Output")
    ax1.plot(time, active_signal, label = "Active Filter Output")
    ax1.set_ylabel('Signal Amplitude')
    ax1.legend()
    
    
    speed_color = 'tab:red' 
    ax2 = ax1.twinx()
    ax2.set_ylabel('Speed (km/h)', color=speed_color)
    ax2.tick_params(axis='y', labelcolor=speed_color)
    ax2.plot(time, speed_data, color=speed_color, label = "Speed")
    ax2.legend()
    
    if(show_cpp):
        plt.plot(time,cppProcValues)
        # plt.plot(time,cppiirValues)

        
    buton_set = False
    for i in range(len(prep_data)):

        if(buton_data[i] == 1) and (not buton_set):            
            plt.axvline(time[i], color='green')
            buton_set = True
        if(buton_data[i] == 0) and buton_set:
            buton_set = False
            
    plt.show()


if __name__ == "__main__":
    main()
    
    
    
    
    

# import numpy as np
# import matplotlib.pyplot as plt
# import math
# #Test QPSK Signal
# num_symbols = 256*10240
# sps = 2

# x_int = np.random.randint(0, 4, num_symbols) # 0 to 3

# x_int = np.repeat(x_int, sps, axis=0)

# x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
# x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
# x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols

# # Create our raised-cosine filter
# num_taps = 101
# beta = 0.35
# Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
# t = np.arange(-51, 52) # remember it's not inclusive of final number
# h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

# # Filter our signal, in order to apply the pulse shaping
# x_shaped = np.convolve(x_symbols, h)

# n = (np.random.randn(len(x_shaped)) + 1j*np.random.randn(len(x_shaped)))/np.sqrt(2) # AWGN with unity power
# noise_power = 0.001
# r = x_shaped + n * np.sqrt(noise_power)

# samples = r
#Function


"""
def create_wave():
    sec = 50
    time = np.linspace(0,sec,1000,endpoint=True)
    signal_freq = 2
    signal1 = 10*np.sin(2*np.pi*signal_freq*time)    
    signal2 = 10*np.sin(2*np.pi*signal_freq*3*time)
    signal3 = 10*np.sin(2*np.pi*signal_freq*10*time)
    
    outSignal = signal1+signal2+signal3
    
    return outSignal

def fft_intensity_plot(samples: np.ndarray, fft_len: int = 256, fft_div: int = 2, mag_steps: int = 100, cmap: str = 'plasma'):
    
    num_ffts = math.floor(len(samples)/fft_len)
    
    fft_array = []
    for i in range(num_ffts):
        temp = np.fft.fftshift(np.fft.fft(samples[i*fft_len:(i+1)*fft_len]))
        temp_mag = 20.0 * np.log10(np.abs(temp))
        fft_array.append(temp_mag)
        
    max_mag = np.amax(fft_array)
    min_mag = np.abs(np.amin(fft_array))
    
    norm_fft_array = fft_array
    for i in range(num_ffts):
        norm_fft_array[i] = (fft_array[i]+(min_mag))/(max_mag+(min_mag)) 
        
    mag_step = 1/mag_steps

    hitmap_array = np.random.random((mag_steps+1,int(fft_len/fft_div)))*np.exp(-10)

    for i in range(num_ffts):
        for m in range(fft_len):
            hit_mag = int(norm_fft_array[i][m]/mag_step)
            hitmap_array[hit_mag][int(m/fft_div)] = hitmap_array[hit_mag][int(m/fft_div)] + 1

    hitmap_array_db = 20.0 * np.log10(hitmap_array+1)
    
    figure, axes = plt.subplots()
    axes.imshow(hitmap_array_db, origin='lower', cmap=cmap, interpolation='bilinear')
    
    return(figure)


"""
