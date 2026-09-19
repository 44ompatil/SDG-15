
# AI-Based Invasive Animal Detection & Alert System

## 1. Project Idea

### Problem

Invasive alien species can negatively affect local ecosystems, vegetation, and agricultural areas. Detecting them early is difficult when monitoring is done manually.

### Solution

We propose a **network of AI-powered ESP32-CAM nodes** that monitor a selected campus/agricultural area.

The camera captures an animal image, an **ML model identifies the target invasive species**, and the node sends an alert to other connected nodes or a central monitoring system.

```text
Animal enters monitored area
          ↓
     ESP32-CAM
          ↓
     Image captured
          ↓
      ML model
          ↓
    Animal identified
          ↓
 Target invasive species?
       ↙          ↘
     No            Yes
     ↓              ↓
   Ignore      Send alert
                    ↓
          Other connected nodes
                    ↓
             Warning / Action
```

---

# 2. Main Features

| Feature                        | Description                                                               |
| ------------------------------ | ------------------------------------------------------------------------- |
| **Animal Detection**           | Detects animals entering the monitored area using the onboard camera.     |
| **ML-Based Identification**    | Classifies the detected animal using a trained ML model.                  |
| **Invasive Species Detection** | Specifically identifies the selected invasive/prioritized species.        |
| **Real-Time Alert**            | Sends an alert when the target species is detected.                       |
| **Multi-Node Communication**   | Multiple ESP32 nodes can monitor different locations.                     |
| **Local Warning**              | Buzzer/LED can be activated at the detecting node.                        |
| **Event Logging**              | Stores species, time, node/location and detection information.            |
| **Image Storage**              | Captures and stores detected-animal images for verification and analysis. |
| **Scalable Monitoring**        | Additional nodes can be added to cover a larger area.                     |

---

# 3. Hardware Components

| Component                  | Purpose                                        |
| -------------------------- | ---------------------------------------------- |
| **ESP32-CAM**              | Main monitoring and camera node.               |
| **ESP32**                  | Additional communication/alert nodes.          |
| **Camera**                 | Captures images for ML-based identification.   |
| **Buzzer**                 | Produces an audible warning after detection.   |
| **LEDs**                   | Indicates system status and detected events.   |
| **MicroSD Card**           | Stores images and detection logs.              |
| **Wi-Fi / ESP-NOW**        | Enables communication between nodes.           |
| **Battery / Power Supply** | Powers the monitoring nodes.                   |
| **Voltage Regulator**      | Provides suitable voltage to the electronics.  |
| **Protective Enclosure**   | Protects the system during outdoor deployment. |

---

# 4. Software & Technology

| Technology                         | Purpose                                                 |
| ---------------------------------- | ------------------------------------------------------- |
| **Python**                         | Dataset preparation and ML model training.              |
| **Machine Learning**               | Classification of target animal species.                |
| **Computer Vision**                | Processing camera images.                               |
| **TensorFlow Lite / Edge Impulse** | Deploying the trained model for edge inference.         |
| **ESP32 Firmware**                 | Camera control, ML inference, alerts and communication. |
| **Wi-Fi / ESP-NOW**                | Communication between monitoring nodes.                 |
| **Data Logging**                   | Recording detection events for analysis.                |

---

# 5. System Architecture

```text
        ┌─────────────┐
        │   Node 1    │
        │ ESP32-CAM   │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │   Node 2    │
        │ ESP32-CAM   │
        └──────┬──────┘
               │
               │ Network
               │
        ┌──────▼──────┐
        │   Node 3    │
        │ ESP32-CAM   │
        └──────┬──────┘
               │
               ▼
      ┌──────────────────┐
      │ Central Receiver │
      │ / Alert System   │
      └────────┬─────────┘
               │
               ▼
        Warning / Action
```

---

# 6. SDG Alignment

## **SDG 15 — Life on Land**

### **Target 15.8**

**Target:** Prevent the introduction and significantly reduce the impact of invasive alien species on land and water ecosystems, including controlling or eradicating priority species.

### How our project supports it

The system provides an **early detection mechanism** for a selected invasive alien species.

```text
Invasive species enters area
          ↓
AI detects and identifies it
          ↓
Alert generated
          ↓
Location/time recorded
          ↓
Early intervention becomes possible
          ↓
Reduced spread / impact
```

The important point is that the project should **specifically identify an invasive alien species**, rather than simply detecting any wild animal.

---

## **Target 15.9**

**Target:** Integrate ecosystem and biodiversity values into national and local planning and development processes.

### How our project supports it

The system creates useful local biodiversity data such as:

* species detected
* detection frequency
* location of sightings
* time of occurrence
* recurring hotspots
* changes in occurrence over time

This data can help campus/local authorities make better decisions about:

* green-zone management
* vegetation planning
* protected areas
* monitoring zones
* invasive-species control
* biodiversity management

So the project isn't only:

> **"Detect an animal."**

It becomes:

> **"Collect biodiversity information that can support better local ecosystem management."**

---

# 7. Simple Project Pitch

> **“We are developing an AI-powered network of ESP32-CAM nodes that detects selected invasive alien species in real time, sends alerts across connected nodes, and records their location and occurrence data. The system supports early detection and management of invasive species under SDG Target 15.8, while the collected biodiversity data can support better local ecosystem planning under Target 15.9.”**


