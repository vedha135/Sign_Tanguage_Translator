import cv2
import time
import os
import tempfile

# Optional hardware/software libraries
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from RPLCD.i2c import CharLCD
    LCD_AVAILABLE = True
except ImportError:
    LCD_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False


# ---------------- CONFIGURATION ----------------

CAMERA_INDEX = 0

# Raspberry Pi GPIO pin numbers (BCM mode)
BUTTON_PIN = 17

# LCD I2C configuration
LCD_ADDRESS = 0x27

# GSM configuration
GSM_PORT = "/dev/serial0"
GSM_BAUD_RATE = 9600

# Enter the recipient phone number only when GSM is configured.
# Leave empty to disable SMS.
PHONE_NUMBER = ""

# Gesture labels
GESTURES = {
    "OPEN_PALM": "HELLO",
    "FIST": "STOP",
    "THUMB_UP": "YES",
    "PEACE": "PEACE"
}


# ---------------- HARDWARE SETUP ----------------

lcd = None
gsm = None
button_enabled = False


def setup_hardware():
    global lcd, gsm, button_enabled

    if GPIO_AVAILABLE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            button_enabled = True
            print("Push button ready")
        except Exception as error:
            print("Button setup error:", error)

    if LCD_AVAILABLE:
        try:
            lcd = CharLCD(
                "PCF8574",
                address=LCD_ADDRESS,
                port=1,
                cols=16,
                rows=2
            )
            lcd.clear()
            lcd.write_string("Sign Translator")
            print("LCD ready")
        except Exception as error:
            print("LCD setup error:", error)
            lcd = None

    if SERIAL_AVAILABLE and PHONE_NUMBER:
        try:
            gsm = serial.Serial(
                GSM_PORT,
                GSM_BAUD_RATE,
                timeout=1
            )
            print("GSM module ready")
        except Exception as error:
            print("GSM setup error:", error)
            gsm = None


# ---------------- OUTPUT FUNCTIONS ----------------

def display_text(message):
    print("Recognized:", message)

    if lcd:
        try:
            lcd.clear()
            lcd.write_string(message[:16])
        except Exception as error:
            print("LCD error:", error)


def speak_text(message):
    if not SPEECH_AVAILABLE:
        print("Speech libraries are not installed")
        return

    try:
        speech = gTTS(text=message, lang="en")

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        )
        audio_path = temp_file.name
        temp_file.close()

        speech.save(audio_path)

        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.quit()
        os.remove(audio_path)

    except Exception as error:
        print("Speech error:", error)


def send_sms(message):
    if gsm is None or not PHONE_NUMBER:
        print("SMS disabled: configure GSM and phone number")
        return

    try:
        gsm.write(b"AT\r")
        time.sleep(1)

        gsm.write(b"AT+CMGF=1\r")
        time.sleep(1)

        command = f'AT+CMGS="{PHONE_NUMBER}"\r'
        gsm.write(command.encode())
        time.sleep(1)

        gsm.write(message.encode())
        gsm.write(bytes([26]))
        time.sleep(3)

        print("SMS command sent")

    except Exception as error:
        print("GSM SMS error:", error)


# ---------------- GESTURE RECOGNITION ----------------

def recognize_gesture(hand_landmarks, handedness):
    points = hand_landmarks.landmark

    # Finger tip and joint positions
    index_up = points[8].y < points[6].y
    middle_up = points[12].y < points[10].y
    ring_up = points[16].y < points[14].y
    pinky_up = points[20].y < points[18].y

    # Thumb direction depends on left/right hand
    if handedness == "Right":
        thumb_up = points[4].x < points[3].x
    else:
        thumb_up = points[4].x > points[3].x

    # Open palm: all fingers extended
    if (
        index_up and middle_up and ring_up
        and pinky_up and thumb_up
    ):
        return "OPEN_PALM"

    # Fist: all fingers folded
    if not (
        index_up or middle_up or ring_up
        or pinky_up or thumb_up
    ):
        return "FIST"

    # Thumb up
    if (
        thumb_up and not index_up
        and not middle_up and not ring_up
        and not pinky_up
    ):
        return "THUMB_UP"

    # Peace sign: index and middle fingers extended
    if (
        index_up and middle_up
        and not ring_up and not pinky_up
    ):
        return "PEACE"

    return None


# ---------------- MAIN PROGRAM ----------------

def main():
    if not MEDIAPIPE_AVAILABLE:
        print("Please install MediaPipe before running this program.")
        return

    setup_hardware()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Camera could not be opened")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    capture_enabled = True
    previous_button_state = 1
    last_button_time = 0
    last_gesture = None
    last_gesture_time = 0

    print("Sign Language Translator Started")
    print("Press Q to quit")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read camera frame")
                break

            # Check push button
            if button_enabled:
                current_state = GPIO.input(BUTTON_PIN)

                if (
                    current_state == 0
                    and previous_button_state == 1
                    and time.time() - last_button_time > 0.4
                ):
                    capture_enabled = not capture_enabled
                    last_button_time = time.time()

                previous_button_state = current_state

            if capture_enabled:
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_index, hand_landmarks in enumerate(
                        results.multi_hand_landmarks
                    ):
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                        handedness = results.multi_handedness[
                            hand_index
                        ].classification[0].label

                        gesture = recognize_gesture(
                            hand_landmarks,
                            handedness
                        )

                        if gesture in GESTURES:
                            current_time = time.time()

                            # Avoid repeated output for the same gesture
                            if (
                                gesture != last_gesture
                                or current_time - last_gesture_time > 3
                            ):
                                word = GESTURES[gesture]

                                display_text(word)
                                speak_text(word)
                                send_sms(word)

                                last_gesture = gesture
                                last_gesture_time = current_time

                            cv2.putText(
                                frame,
                                GESTURES[gesture],
                                (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 255, 0),
                                3
                            )

            status = (
                "Capture ON"
                if capture_enabled
                else "Capture OFF"
            )

            cv2.putText(
                frame,
                status,
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            cv2.imshow(
                "Sign Language Translator",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()

    if lcd:
        lcd.clear()

    if gsm:
        gsm.close()

    if GPIO_AVAILABLE and button_enabled:
        GPIO.cleanup()

    print("Sign Language Translator Stopped")


if __name__ == "__main__":
    main()