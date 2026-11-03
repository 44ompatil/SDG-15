# ESP32-S3 wildlife camera

The ESP32-S3 CAM captures JPEGs when the HC-SR501 PIR sensor detects motion
and sends them over Wi-Fi to the host computer. The host runs the existing
PyTorch wildlife classifier, saves each image under `hardware/capImages/`,
appends predictions to `hardware/detections.csv`, and serves the live dashboard.
The ESP32 no longer runs model inference, which removes the roughly 8.5-second
on-device inference delay. The ESP32 and host computer must remain powered on
and connected to the same network.

## Hardware wiring

The board is an ESP32-S3 N16R8 with OV3660. Verify the camera pin mapping in
`include/board_config.h` against your board's schematic before flashing.

| HC-SR501 | ESP32-S3 CAM |
| --- | --- |
| VCC | 5V |
| GND | GND |
| OUT | GPIO 2 (`PIR_GPIO` in `include/board_config.h`) |

Connect each indicator LED's anode to the specified GPIO through its own
220-330 ohm series resistor; connect each cathode to GND. Use a separate
resistor for every LED.

| Indicator / module | ESP32-S3 CAM connection |
| --- | --- |
| Green LED (no/low threat) anode | GPIO 1 (`LED_LOW_THREAT_GPIO`) through a 220-330 ohm resistor |
| Yellow LED (medium threat) anode | GPIO 14 (`LED_MEDIUM_THREAT_GPIO`) through a 220-330 ohm resistor |
| Red LED (high threat) anode | GPIO 21 (`LED_HIGH_THREAT_GPIO`) through a 220-330 ohm resistor |
| All LED cathodes | GND |
| Active buzzer positive | External 5V supply positive |
| Active buzzer negative | NPN transistor collector |
| NPN transistor emitter | GND |
| GPIO 3 (`BUZZER_GPIO`) | NPN base through a 1-4.7 kohm resistor |
| NPN base | GND through a 10 kohm pull-down resistor |

Use a small NPN transistor (for example, 2N2222) to switch the buzzer; do not
power a buzzer directly from an ESP32 GPIO. Connect the ESP32, PIR, transistor,
and buzzer supply grounds together. For a magnetic/inductive buzzer, place a
flyback diode across it (cathode to the positive supply). Use a 3.3V-compatible
active buzzer if powering it from 3.3V, or a separately rated supply as shown.
GPIO 3 is a boot-strapping pin on the ESP32-S3: the base pull-down must keep the
transistor off during reset, and you must verify that this GPIO is exposed and
safe to use on your exact board. If not, select another unused, non-strapping
GPIO and change `BUZZER_GPIO` in `include/board_config.h`.

Share ground between the PIR and ESP32. ESP32-S3 GPIOs are **not 5V tolerant**;
verify that the PIR OUT voltage is 3.3V before connecting it. Verify every
selected GPIO is exposed on your board and does not overlap a camera signal.

The firmware ignores PIR events during the sensor's 30-second warm-up. After
each uploaded frame is classified by the host, the receiver sends the threat
level back to the ESP32 and exactly one LED indicates that result: green for
no alert, yellow for `wild_boar` or `nilgai`, and red for `leopard`, `tiger`,
or `hyena`. Classifications below the 0.70 confidence threshold, including
`human`, show green. The LEDs remain on the most recent successfully returned
classification; they are off until the first classification. The buzzer
chirps for 100 ms once per PIR motion event, and only if a high-threat
classification is returned. Upload/setup errors are reported in the serial
monitor, not represented by the threat LEDs. The firmware posts a motion alert
and begins sequential JPEG captures while the PIR output remains HIGH.

## Configure the host receiver

Use the repository's Python environment (Python 3.12 or newer, as specified by
`pyproject.toml`) with its PyTorch, torchvision, Pillow, and NumPy dependencies
installed. The default checkpoint is `ml/models/best_model.pt`.

If the environment is not set up yet, run this from the repository root:

```powershell
uv sync --python 3.12
```

1. Copy `include/secrets.example.h` to `include/secrets.h`.
2. Set the same long random upload token in `include/secrets.h` and the host's
   `SDG15_UPLOAD_TOKEN` environment variable. Keep `secrets.h` private; it is
   ignored by Git.
3. Set the ESP32 Wi-Fi credentials and upload URL in `include/secrets.h`. The
   URL must use the host computer's LAN address and end with `/capture`, for
   example `http://192.168.1.10:8080/capture`.
4. Start the receiver from the repository root:

   ```powershell
   $env:SDG15_UPLOAD_TOKEN = "the-same-long-random-token"
   .venv\Scripts\python.exe hardware\host_receiver.py --host 0.0.0.0 --port 8080 --device cpu
   ```

   Allow inbound TCP port 8080 through the host's private-network firewall.
   The receiver uses plain HTTP; only use it on a trusted local network. To use
   a different checkpoint, pass `--checkpoint path\to\checkpoint.pt`. Device
   can be `cpu`, `cuda`, or `auto`.

Open `http://127.0.0.1:8080/` on the host computer for the latest PIR-triggered
image, predicted class, confidence, alert flags, and class scores. The dashboard
refreshes as new images arrive; it is not an MJPEG video stream. The detection
CSV includes image path, capture time, class scores, model version, host
inference duration, PIR/camera information, and alert flags. Motion-start
events are logged separately in `hardware/motion_events.csv`.

Keep the receiver process running while using the dashboard. If the dashboard
cannot connect, check that the receiver started successfully and that the
browser is open to the receiver URL above, not the HTML file directly. Each
image path in the CSV must also exist in the configured JPEG directory
(`hardware/capImages` by default). The CSV does not contain image data, and the
ESP32 releases each camera frame after upload; if an older JPEG is missing,
it cannot be reconstructed from the CSV. Start the receiver before triggering
motion to save and display new captures. If JPEGs are available in another
folder, pass that folder with `--image-dir`.

The receiver acknowledges each capture after inference and saving. On the ESP32
serial monitor, look for `MOTION ALERT`, `Uploaded capture`, or the associated
upload error. Classification results are available in the dashboard and CSV.

## Accuracy and speed

Moving inference to the host addresses the ESP32's measured 8.5-second model
inference delay. On the available CPU, warmed model calls measured 16-21 ms
after an initial 148 ms cold call; capture decoding, upload, and persistence add
more time. Actual latency depends on the host, network, and image size.

This change does **not** improve the model's accuracy. The existing checkpoint
scored 83.5% accuracy on its held-out test set, which is not a guarantee of
field performance. It was trained on the available dataset, not representative
camera-trap images from the deployment site. Increasing field precision and
recall requires representative, correctly labeled examples and validation;
retraining the same data cannot substantiate a high-accuracy claim. The
classifier's alert threshold remains 0.70.

## Build and flash firmware

Install PlatformIO Core or the VS Code PlatformIO extension. From `hardware`,
build and flash the firmware (replace `COM5` with the ESP32's serial port):

```powershell
pio run
pio run --target upload --upload-port COM5
pio device monitor --port COM5 --baud 115200
```

The firmware uses the existing partition table to preserve the device's flash
layout. It does not load or require a model partition; upload the firmware only.
