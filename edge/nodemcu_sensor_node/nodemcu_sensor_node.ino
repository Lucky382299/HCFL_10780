/*
 * NodeMCU (ESP8266) sensor node: MQ-135 (analog), PMS5003 (UART), SHT25 (I2C).
 * Samples at 1 Hz, applies light calibration/filtering, and publishes a
 * timestamped JSON packet over MQTT to the Raspberry Pi edge client.
 *
 * Libraries (Arduino Library Manager): PubSubClient, ArduinoJson (v6+), PMS Library (fu-hsi),
 * NTPClient. Board: "NodeMCU 1.0 (ESP-12E)".
 *
 * Wiring: MQ-135 AOUT -> A0 (through a 3.3 V divider)
 *         PMS5003 TX  -> D7 (SoftwareSerial RX), RX <- D8
 *         SHT25 SDA   -> D2, SCL -> D1 (I2C addr 0x40)
 */
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <SoftwareSerial.h>
#include <PMS.h>
#include <Wire.h>
#include <WiFiUdp.h>
#include <NTPClient.h>

const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* MQTT_HOST = "192.168.1.50";   // Raspberry Pi IP (Mosquitto broker)
const uint16_t MQTT_PORT = 1883;
const char* NODE_ID   = "node01";         // unique per NodeMCU
const unsigned long PERIOD_MS = 1000;     // 1 sample per second

WiFiClient wifi;
PubSubClient mqtt(wifi);
SoftwareSerial pmsSerial(D7, D8);
PMS pms(pmsSerial);
PMS::DATA pmsData;
WiFiUDP ntpUDP;
NTPClient ntp(ntpUDP, "pool.ntp.org", 0, 60000);

const uint8_t SHT25_ADDR = 0x40;
float ema_mq = NAN, ema_pm25 = NAN;        // light EMA filtering
const float EMA_A = 0.3;
uint32_t seq = 0;

float sht25Read(uint8_t cmd) {             // 0xE3 = temperature, 0xE5 = humidity (hold master)
  Wire.beginTransmission(SHT25_ADDR); Wire.write(cmd);
  if (Wire.endTransmission() != 0) return NAN;
  delay(cmd == 0xE3 ? 85 : 30);
  Wire.requestFrom(SHT25_ADDR, (uint8_t)3);
  if (Wire.available() < 3) return NAN;
  uint16_t raw = ((uint16_t)Wire.read() << 8) | Wire.read(); Wire.read();  // drop CRC
  raw &= ~0x0003;
  return (cmd == 0xE3) ? (-46.85 + 175.72 * raw / 65536.0) : (-6.0 + 125.0 * raw / 65536.0);
}

float mq135ppm(int adc) {                  // calibrate R0 in clean air for your sensor
  const float RL = 10.0, R0 = 76.63;
  float v = adc * (3.3 / 1023.0);
  if (v <= 0.01) return NAN;
  float rs = (3.3 - v) * RL / v;
  return 116.6020682 * pow(rs / R0, -2.769034857);
}

float ema(float prev, float x) { return isnan(prev) ? x : EMA_A * x + (1 - EMA_A) * prev; }

void connectAll() {
  while (WiFi.status() != WL_CONNECTED) { delay(250); }
  while (!mqtt.connected()) {
    if (!mqtt.connect(NODE_ID)) delay(1000);
  }
}

void setup() {
  Serial.begin(115200);
  pmsSerial.begin(9600);
  pms.passiveMode();
  Wire.begin(D2, D1);
  WiFi.mode(WIFI_STA); WiFi.begin(WIFI_SSID, WIFI_PASS);
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  connectAll();
  ntp.begin();
}

void loop() {
  static unsigned long last = 0;
  if (!mqtt.connected()) connectAll();
  mqtt.loop(); ntp.update();
  if (millis() - last < PERIOD_MS) return;
  last = millis();

  float co2 = mq135ppm(analogRead(A0));
  pms.requestRead();
  float pm25 = NAN, pm10 = NAN;
  if (pms.readUntil(pmsData, 900)) { pm25 = pmsData.PM_AE_UG_2_5; pm10 = pmsData.PM_AE_UG_10_0; }
  float t = sht25Read(0xE3), rh = sht25Read(0xE5);
  ema_mq = ema(ema_mq, co2); ema_pm25 = ema(ema_pm25, pm25);

  StaticJsonDocument<256> doc;
  doc["node"] = NODE_ID; doc["seq"] = seq++;
  doc["ts_ms"] = (uint64_t)ntp.getEpochTime() * 1000ULL + (millis() % 1000);
  JsonArray x = doc.createNestedArray("x");
  x.add(ema_mq); x.add(ema_pm25); x.add(pm10); x.add(t); x.add(rh);
  doc["rssi"] = WiFi.RSSI();
  char buf[256]; size_t n = serializeJson(doc, buf);
  String topic = String("hcfl/sensor/") + NODE_ID;
  mqtt.publish(topic.c_str(), buf, n);
}
