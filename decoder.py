from concurrent.futures import ThreadPoolExecutor
from rtlsdr.rtlsdraio import RtlSdrAio
from sounddevice import OutputStream
from keyboard import add_hotkey
from scipy import signal
from sys import argv

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import time as t
import numpy as np
import asyncio
import logging

# Constants
syndrome = [383, 14, 303, 663, 748]
offset_pos = [0, 1, 2, 3, 2]
offset_word = [252, 408, 360, 436, 848]

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(lineno)d]: %(message)s",
    "%m/%d/%Y %I:%M:%S %p"
)

if len(argv) > 1:
    levels = []
    for value in argv[1:]:
        if value == "debug":
            levels.append(logging.DEBUG)
        elif value == "info":
            levels.append(logging.INFO)
        else:
            levels.append(float(value))

    if len(levels) == 1:
        stream_level = levels[0]
        file_level = logging.INFO
    elif len(levels) > 1:
        stream_level = levels[0]
        file_level = levels[1]
    else:
        stream_level = logging.INFO
        file_level = logging.INFO
else:
    stream_level = logging.INFO
    file_level = logging.INFO

stream_handler = logging.StreamHandler()
stream_handler.setLevel(stream_level)
stream_handler.setFormatter(formatter)

file_handler = logging.FileHandler("latest.log")
file_handler.setLevel(file_level)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def iq_correction(samples: np.ndarray):
    # Remove DC and calculate input power
    centered_samples = samples - np.mean(samples)
    input_power = np.var(centered_samples)

    # Calculate scaling factor for Q
    q_amplitude = np.sqrt(2 * np.mean(samples.imag ** 2))

    # Normalize Q component
    normalized_samples = samples / q_amplitude

    i_samples, q_samples = normalized_samples.real, normalized_samples.imag

    # Estimate alpha and sin_phi
    alpha_est = np.sqrt(2 * np.mean(i_samples ** 2))
    sin_phi_est = (2 / alpha_est) * np.mean(i_samples * q_samples)

    # Estimate cos_phi
    cos_phi_est = np.sqrt(1 - sin_phi_est ** 2)

    # Apply phase and amplitude correction
    i_new = (1 / alpha_est) * i_samples
    q_new = (-sin_phi_est / alpha_est) * i_samples + q_samples

    # Corrected signal
    corrected_samples = (i_new + 1j * q_new) / cos_phi_est

    # Calculate and print phase and amplitude errors
    # phase_error = np.arccos(cos_phi_est) * 180 / np.pi
    # amplitude_error = 20 * np.log10(alpha_est)
    #
    # logger.debug(f"Phase Error: {phase_error} degrees")
    # logger.debug(f"Amplitude Error: {amplitude_error} dB")

    return corrected_samples * np.sqrt(input_power / np.var(corrected_samples))


# see Annex B, page 64 of the standard
def calc_syndrome(x, mlen):
    reg = 0
    p_len = 10
    for ii in range(mlen, 0, -1):
        reg = (reg << 1) | ((x >> (ii - 1)) & 0x01)
        if reg & (1 << p_len):
            reg = reg ^ 0x5B9
    for ii in range(p_len, 0, -1):
        reg = reg << 1
        if reg & (1 << p_len):
            reg = reg ^ 0x5B9
    return reg & ((1 << p_len) - 1)  # select the bottom p_len bits of reg


def calc_freq_offset(output: np.ndarray, freq_log: list):
    subset = output[0:2000]

    scores = []
    for i in range(int(len(subset) / 10 - 2)):
        x = np.real(subset[i * 10:(i + 2) * 10])
        y = np.imag(subset[i * 10:(i + 2) * 10])

        features = StandardScaler().fit_transform(np.column_stack((x, y)))
        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
        kmeans.fit(features)

        scores.append(kmeans.inertia_)

    score_1 = np.percentile(np.array(scores), 25)

    freq_log = np.array(freq_log)
    freq_offset = np.mean(freq_log)
    score_2 = np.mean(np.abs(freq_offset - np.mean(freq_offset)))

    score = np.sqrt((score_1**2) + (score_2**2))
    logger.debug(f"RDS decoding score is {score}")

    return freq_offset


def decode_rds(samples: np.ndarray, sample_rate: int, taps: np.ndarray, freq_offset: int = 0):
    # FM Demodulation
    x = 0.5 * np.angle(samples[0:-1] * np.conj(samples[1:]))

    # Frequency shift
    time = np.arange(len(x)) / sample_rate
    x = x * np.exp(2j * np.pi * -(57e3 + freq_offset) * time)

    # Filter to isolate RDS
    x = np.convolve(x, taps, "valid")

    # Decimate by 10
    x = signal.decimate(x, 10)
    # sample_rate = 25e3

    # Resample to 19 kHz
    x = signal.resample_poly(x, 19, 25)
    sample_rate = 19e3

    # Time synchronization (symbol level)
    samples = x.astype(np.complex64)
    samples_interpolated = signal.resample_poly(samples, 16, 1)
    sps = 16
    mu = 0.01  # initial estimate of phase of sample
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)
    i_in = 0  # input samples index
    i_out = 2  # output index (let first two outputs be 0)
    while i_out < len(samples) and i_in + 16 < len(samples):
        out[i_out] = samples_interpolated[i_in * 16 + int(mu * 16)]  # grab what we think is the "best" sample
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j * int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out - 2]) * np.conj(out[i_out - 1])
        y = (out[i_out] - out[i_out - 2]) * np.conj(out_rail[i_out - 1])
        mm_val = np.real(y - x)
        mu += sps + 0.01 * mm_val
        i_in += int(np.floor(mu))  # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu)  # remove the integer part of mu
        i_out += 1  # increment output index
    x = out[2:i_out]  # remove the first two, and anything after i_out (that was never filled out)

    # Fine frequency synchronization
    samples = x  # for the sake of matching the sync chapter
    n = len(samples)
    phase = 0
    freq = 0
    # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
    alpha = 4.25
    beta = 0.0008
    out = np.zeros(n, dtype=np.complex64)
    freq_log = []
    for i in range(n):
        # adjust the input sample by the inverse of the estimated phase offset
        out[i] = samples[i] * np.exp(-1j * phase)
        # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
        error = np.real(out[i]) * np.imag(out[i])

        # Advance the loop (recalc phase and freq offset)
        freq += (beta * error)
        freq_log.append(freq * sample_rate / (2 * np.pi))  # convert from angular velocity to Hz for logging
        phase += freq + (alpha * error)

        # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
        while phase >= 2 * np.pi:
            phase -= 2 * np.pi
        while phase < 0:
            phase += 2 * np.pi
    x = out

    # Demod BPSK
    bits = (np.real(x) > 0).astype(int)  # 1's and 0's

    # Differential decoding, so that it doesn't matter whether our BPSK was 180 degrees rotated without us
    bits = (bits[1:] - bits[0:-1]) % 2
    bits = bits.astype(np.uint8)  # for decoder

    # Initialize all the working vars we'll need during the loop
    synced = False
    presync = False

    wrong_blocks_counter = 0
    blocks_counter = 0
    group_good_blocks_counter = 0

    reg = np.uint32(0)  # Was unsigned long in C++ (64 bits), but numpy doesn't support bitwise ops of uint64

    lastseen_offset_counter = 0
    lastseen_offset = 0

    # The synchronization process is described in Annex C, page 66 of the standard
    bytes_out = []

    for i in range(len(bits)):
        # In C++ reg doesn't get init, so it will be random at first, for ours its 0s
        # It was also an unsigned long but never seemed to get anywhere near the max value
        # bits are either 0 or 1
        reg = np.bitwise_or(np.left_shift(reg, 1),
                            bits[i])  # Reg contains the last 26 rds bits. These are both bitwise ops
        if not synced:
            reg_syndrome = calc_syndrome(reg, 26)
            for j in range(5):
                if reg_syndrome == syndrome[j]:
                    if not presync:
                        lastseen_offset = j
                        lastseen_offset_counter = i
                        presync = True
                    else:
                        if offset_pos[lastseen_offset] >= offset_pos[j]:
                            block_distance = offset_pos[j] + 4 - offset_pos[lastseen_offset]
                        else:
                            block_distance = offset_pos[j] - offset_pos[lastseen_offset]
                        if (block_distance * 26) != (i - lastseen_offset_counter):
                            presync = False
                        else:
                            logger.info('Sync State Detected')
                            wrong_blocks_counter = 0
                            blocks_counter = 0
                            block_bit_counter = 0
                            block_number = (j + 1) % 4
                            group_assembly_started = False
                            synced = True
                        break  # Syndrome found, no more cycles

        else:  # Synced
            # Wait until 26 bits enter the buffer
            if block_bit_counter < 25:
                block_bit_counter += 1
            else:
                good_block = False
                dataword = (reg >> 10) & 0xffff
                block_calculated_crc = calc_syndrome(dataword, 16)
                checkword = reg & 0x3ff
                if block_number == 2:  # Manage special case of C or C's offset word
                    block_received_crc = checkword ^ offset_word[block_number]
                    if block_received_crc == block_calculated_crc:
                        good_block = True
                    else:
                        block_received_crc = checkword ^ offset_word[4]
                        if block_received_crc == block_calculated_crc:
                            good_block = True
                        else:
                            wrong_blocks_counter += 1
                            good_block = False
                else:
                    block_received_crc = checkword ^ offset_word[block_number]  # Bitwise xor
                    if block_received_crc == block_calculated_crc:
                        good_block = True
                    else:
                        wrong_blocks_counter += 1
                        good_block = False

                # Done checking CRC
                if block_number == 0 and good_block:
                    group_assembly_started = True
                    group_good_blocks_counter = 1
                    bytes_array = bytearray(8)  # 8 bytes filled with 0s
                if group_assembly_started:
                    if not good_block:
                        group_assembly_started = False
                    else:
                        # Raw data bytes, as received from RDS. 8 info bytes,
                        # followed by 4 RDS offset chars: ABCD/ABcD/EEEE (in US)
                        # which we leave out here
                        # RDS information words
                        # Block_number is either 0,1,2,3 so this is how we fill out the 8 bytes
                        bytes_array[block_number * 2] = (dataword >> 8) & 255
                        bytes_array[block_number * 2 + 1] = dataword & 255
                        group_good_blocks_counter += 1
                        # print('group_good_blocks_counter:', group_good_blocks_counter)
                    if group_good_blocks_counter == 5:
                        # print(bytes_array)
                        bytes_out.append(bytes_array)  # List of len-8 lists of bytes
                block_bit_counter = 0
                block_number = (block_number + 1) % 4
                blocks_counter += 1
                if blocks_counter == 50:
                    if wrong_blocks_counter > 35:  # This many wrong blocks must mean we lost sync
                        logger.info(f"Lost Sync (Got {wrong_blocks_counter} bad blocks on {blocks_counter} total)")
                        synced = False
                        presync = False
                    else:
                        logger.info(f"Still Sync-ed (Got {wrong_blocks_counter} bad blocks on {blocks_counter} total)")
                    blocks_counter = 0
                    wrong_blocks_counter = 0

    # Annex F of RDS Standard Table F.1 (North America) and Table F.2 (Europe)
    # Europe, North America
    pty_table = [
        ["Undefined", "Undefined"],
        ["News", "News"],
        ["Current Affairs", "Information"],
        ["Information", "Sports"],
        ["Sport", "Talk"],
        ["Education", "Rock"],
        ["Drama", "Classic Rock"],
        ["Culture", "Adult Hits"],
        ["Science", "Soft Rock"],
        ["Varied", "Top 40"],
        ["Pop Music", "Country"],
        ["Rock Music", "Oldies"],
        ["Easy Listening", "Soft"],
        ["Light Classical", "Nostalgia"],
        ["Serious Classical", "Jazz"],
        ["Other Music", "Classical"],
        ["Weather", "Rhythm & Blues"],
        ["Finance", "Soft Rhythm & Blues"],
        ["Children’s Programmes", "Language"],
        ["Social Affairs", "Religious Music"],
        ["Religion", "Religious Talk"],
        ["Phone-In", "Personality"],
        ["Travel", "Public"],
        ["Leisure", "College"],
        ["Jazz Music", "Spanish Talk"],
        ["Country Music", "Spanish Music"],
        ["National Music", "Hip Hop"],
        ["Oldies Music", "Unassigned"],
        ["Folk Music", "Unassigned"],
        ["Documentary", "Weather"],
        ["Alarm Test", "Emergency Test"],
        ["Alarm", "Emergency"]
    ]
    pty_locale = 1  # set to 0 for Europe which will use first column instead

    # page 72, Annex D, table D.2 in the standard
    coverage_area_codes = [
        "Local",
        "International",
        "National",
        "Supra-regional",
        "Regional 1",
        "Regional 2",
        "Regional 3",
        "Regional 4",
        "Regional 5",
        "Regional 6",
        "Regional 7",
        "Regional 8",
        "Regional 9",
        "Regional 10",
        "Regional 11",
        "Regional 12"
    ]

    radiotext_ab_flag = 0
    radiotext = [' '] * 65
    first_time = True
    for bytes_array in bytes_out:
        group_0 = bytes_array[1] | (bytes_array[0] << 8)
        group_1 = bytes_array[3] | (bytes_array[2] << 8)
        group_2 = bytes_array[5] | (bytes_array[4] << 8)
        group_3 = bytes_array[7] | (bytes_array[6] << 8)

        # here is what each one means, e.g. RT is radiotext which is the only one we decode here: ["BASIC", "PIN/SL"
        # , "RT", "AID", "CT", "TDC", "IH", "RP", "TMC", "EWS", "___", "___", "___", "___", "EON", "___"]
        group_type = (group_1 >> 12) & 0xf
        ab = (group_1 >> 11) & 0x1  # b if 1, 'a' if 0

        # this is essentially message type, I only see type 0 and 2 in my recording
        logger.debug(f"Group Type: {group_type}")
        logger.debug(f"AB: {ab}")

        program_identification = group_0  # "PI"

        program_type = (group_1 >> 5) & 0x1f  # "PTY"
        pty = pty_table[program_type][pty_locale]

        pi_area_coverage = (program_identification >> 8) & 0xf
        coverage_area = coverage_area_codes[pi_area_coverage]

        pi_program_reference_number = program_identification & 0xff  # just an int

        if first_time:
            logger.info(f"PTY: {pty}")
            logger.info(f"Program: {pi_program_reference_number}")
            logger.info(f"Coverage Area: {coverage_area}")
            first_time = False

        if group_type == 2:
            # when the A/B flag is toggled, flush your current radiotext
            if radiotext_ab_flag != ((group_1 >> 4) & 0x01):
                radiotext = [' '] * 65
            radiotext_ab_flag = (group_1 >> 4) & 0x01
            text_segment_address_code = group_1 & 0x0f
            if ab:
                radiotext[text_segment_address_code * 2] = chr((group_3 >> 8) & 0xff)
                radiotext[text_segment_address_code * 2 + 1] = chr(group_3 & 0xff)
            else:
                radiotext[text_segment_address_code * 4] = chr((group_2 >> 8) & 0xff)
                radiotext[text_segment_address_code * 4 + 1] = chr(group_2 & 0xff)
                radiotext[text_segment_address_code * 4 + 2] = chr((group_3 >> 8) & 0xff)
                radiotext[text_segment_address_code * 4 + 3] = chr(group_3 & 0xff)
            logger.info(f"Radiotext: {''.join(radiotext)}")
        else:
            logger.debug(f"unsupported group_type: {group_type}")

    return out, freq_log


def decode_audio(samples: np.ndarray, sample_rate: float):
    freq_deviation = 75e3
    deemphasis = 75e-6
    decimation = 6

    # FM demodulation
    gain = sample_rate / (2 ** np.pi * freq_deviation)
    demod = gain * np.angle(samples[:-1] * samples.conj()[1:])

    # Decimation to get mono audio
    mono = signal.decimate(demod, decimation, ftype="fir")

    # Bandpass filtering for 19 kHz pilot tone
    filter_cutoff = [18.9e3, 19.1e3]
    taps = signal.firwin(numtaps=101, cutoff=filter_cutoff, fs=sample_rate, pass_zero="bandpass")
    pilot = np.convolve(taps, demod, "valid")
    pilot -= pilot.mean()
    pilot *= 10

    # Bandpass filtering for 38 kHz stereo audio
    filter_cutoff = [22.9e3, 53.1e3]
    taps = signal.firwin(numtaps=101, cutoff=filter_cutoff, fs=sample_rate)
    filter_stereo = np.convolve(taps, demod, "valid")

    # AM coherent demodulation of stereo audio
    carrier = 2 * (2 * pilot ** 2 - 1)
    demod_stereo = filter_stereo * carrier

    # Decimation filter to get stereo audio
    audio_stereo = signal.decimate(demod_stereo, decimation, ftype="fir")

    # De-emphasis filter
    b, a = [1], [deemphasis, 1]

    # transform analog filter into digital filter
    bz, az = signal.bilinear(b, a, fs=sample_rate)

    # apply the de-emphasis filter
    mono = signal.lfilter(bz, az, mono)
    audio_stereo = signal.lfilter(bz, az, audio_stereo)

    pad = np.zeros(np.abs(len(mono) - len(audio_stereo)))
    audio_stereo = np.concatenate((audio_stereo, pad))

    # separate stereo channels
    left = mono + audio_stereo
    right = mono - audio_stereo

    # remove dc offset
    left -= left.mean()
    right -= right.mean()

    # combine left and right channels
    stereo = np.column_stack((left, right))
    return stereo.astype(np.float32)


def play_audio(audio: np.ndarray, stream: OutputStream):
    stream.write(audio)


async def fm_demod(frequency: float):
    sdr = RtlSdrAio(device_index=0)

    center_freq = frequency if np.log10(frequency) > 6 else frequency * 1e6
    sample_rate = int(250e3)
    audio_rate = int(44100 * 0.5)
    rds_duration = 5

    sdr.center_freq = center_freq
    sdr.sample_rate = sample_rate
    sdr.gain = 20.7

    # Create low-pass filter
    taps = signal.firwin(numtaps=101, cutoff=7.5e3, fs=sample_rate)

    # Test the actual audio sample rate
    samples = sdr.read_samples(sdr.sample_rate)
    samples = iq_correction(samples)
    audio = decode_audio(samples, sample_rate)
    actual_rate = len(audio)

    thread_pool = ThreadPoolExecutor(max_workers=4)
    stream = OutputStream(
        samplerate=actual_rate,
        blocksize=actual_rate,
        channels=2,
        dtype=np.float32
    )

    stop_event = asyncio.Event()
    add_hotkey("esc", stop_event.set)
    add_hotkey("q", stop_event.set)

    i = 0
    stream.start()
    freq_offset = 0
    audio = np.zeros((audio_rate, 2), dtype=np.float32)
    output, freq_log = decode_rds(samples, sample_rate, taps)
    async for samples in sdr.stream(sample_rate):
        if stop_event.is_set():
            sdr.stop()
        else:
            start = t.time()
            x = iq_correction(samples)

            future_play = thread_pool.submit(play_audio, audio, stream)
            future_audio = thread_pool.submit(decode_audio, x, sample_rate)

            if i % rds_duration == 0:
                future_adjust = thread_pool.submit(calc_freq_offset, output, freq_log)
                future_rds = thread_pool.submit(decode_rds, x, sample_rate, taps, freq_offset)

            if i % rds_duration == (rds_duration - 1):
                output, freq_log = future_rds.result()
                freq_offset = future_adjust.result()

            audio = future_audio.result()
            future_play.result()

            i += 1

            end = t.time()
            timing = end - start
            if timing > 1:
                keyword = "Bad"
            elif timing > 0.5:
                keyword = "Okay"
            elif timing > 0.1:
                keyword = "Good"
            elif timing < 0.1:
                keyword = "Great"
            else:
                keyword = "Unknown"

            logger.debug(f"Decoding took {round(timing, 4)} ({keyword})")

    await stop_event.wait()

    stream.stop()
    stream.close()

    logger.info("Exiting")
    thread_pool.shutdown(wait=True)


asyncio.run(fm_demod(101.5))
