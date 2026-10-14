//////////////////////////////////////////////////////////////////////////////////////////
//
//    ADS1220 采集 + 癫痫判定，串口输出版（XIAO RP2040 测试用）。
//    引脚与 Brain_ECoG 相同：CS=D3, DRDY=D2, SPI=D8/D9/D10。
//    不含 BLE。确认发作时把治疗时长打到串口。
//
//////////////////////////////////////////////////////////////////////////////////////////

#include "Protocentral_ADS1220.h"
#include <SPI.h>
#include <math.h>

#define SEIZ_D        22.0f
#define SEIZ_T        5
#define SEIZ_R        0.0012f
#define SHORT_WINDOW  20
#define LONG_WINDOW   1000
#define THRESH_LOCK_MS 30000UL

#define BP_F0_HZ      16.75f
#define BP_Q          2.0f

#define PGA_GAIN_SELECT       1
#define SAMPLING_RATE_SELECT  1000
#define INPUT_MODE_SELECT     1
#define FIR_FILTER_ENABLED    1
#define FIR_FILTER_TYPE       FIR_5060

#define VREF         2.048f
#define VFSR         (VREF / (float)PGA_GAIN_SELECT)
#define FULL_SCALE   (((long int)1 << 23) - 1)

// RP2040 的 Arduino 引脚号 = GPIO 号：D3=GPIO29，D2=GPIO28。
// 不能写 3/2，那会变成 MOSI(D10)/SCK(D8)，SPI 读回来全是 0xFF。
#define ADS1220_CS_PIN    D3
#define ADS1220_DRDY_PIN  D2

Protocentral_ADS1220 pc_ads1220;
volatile bool drdyIntrFlag = false;
static uint32_t s_lastPrintMs;

struct BiquadBP {
    float b0, b1, b2, a1, a2;
    float z1, z2;
    void init(float fs, float f0, float Q) {
        const float pi = 3.14159265f;
        float w0 = 2.0f * pi * f0 / fs;
        float c = cosf(w0);
        float s = sinf(w0);
        float a = s / (2.0f * Q);
        float a0u = 1.0f + a;
        b0 = a / a0u;
        b1 = 0;
        b2 = -a / a0u;
        a1 = (-2.0f * c) / a0u;
        a2 = (1.0f - a) / a0u;
        z1 = z2 = 0;
    }
    float process(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};

static BiquadBP s_bp;
static float s_shortBuf[SHORT_WINDOW];
static float s_longBuf[LONG_WINDOW];
static uint16_t s_shortIdx, s_longIdx;
static uint16_t s_shortCount, s_longCount;

static bool s_threshLocked;
static uint32_t s_lockStartMs;
static float s_frozenThresh;

static uint32_t s_seizureWindowStartMs;
static int s_seizurePointCount;
static bool s_therapyActive;
static uint32_t s_therapyStartMs;
static uint32_t s_therapyDurationMs;
static uint16_t s_printDiv;

float convertToMilliV(int32_t i32data) {
    return (float)((double)i32data * (double)VFSR * 1000.0 / (double)FULL_SCALE);
}

static void ringPush(float v) {
    s_shortBuf[s_shortIdx] = v;
    s_shortIdx = (uint16_t)((s_shortIdx + 1) % SHORT_WINDOW);
    if (s_shortCount < SHORT_WINDOW)
        s_shortCount++;

    s_longBuf[s_longIdx] = v;
    s_longIdx = (uint16_t)((s_longIdx + 1) % LONG_WINDOW);
    if (s_longCount < LONG_WINDOW)
        s_longCount++;
}

static float windowVariance(const float *buf, uint16_t n, uint16_t cap) {
    if (n < cap)
        return 0;
    float sum = 0;
    for (uint16_t i = 0; i < cap; i++)
        sum += buf[i];
    float mean = sum / (float)cap;
    float acc = 0;
    for (uint16_t i = 0; i < cap; i++) {
        float d = buf[i] - mean;
        acc += d * d;
    }
    return acc / (float)cap;
}

static float shortSumSqDev(void) {
    if (s_shortCount < SHORT_WINDOW)
        return 0;
    float sum = 0;
    for (uint16_t i = 0; i < SHORT_WINDOW; i++)
        sum += s_shortBuf[i];
    float mean = sum / (float)SHORT_WINDOW;
    float acc = 0;
    for (uint16_t i = 0; i < SHORT_WINDOW; i++) {
        float d = s_shortBuf[i] - mean;
        acc += d * d;
    }
    return acc;
}

static void startTherapy(void) {
    if (s_therapyActive)
        return;
    float sumsq = shortSumSqDev();
    uint32_t dur = (uint32_t)(SEIZ_R * sumsq * 1000.0f);
    if (dur < 100u)
        dur = 100u;
    if (dur > 10000u)
        dur = 10000u;
    s_therapyDurationMs = dur;
    s_therapyActive = true;
    s_therapyStartMs = millis();
    s_seizurePointCount = 0;
    Serial.print(F("THERAPY,"));
    Serial.println(dur);
}

static void runSeizureStep(float *S_out, float *th_out) {
    *S_out = 0;
    *th_out = 0;
    if (s_shortCount < SHORT_WINDOW || s_longCount < LONG_WINDOW)
        return;

    if (s_threshLocked && (millis() - s_lockStartMs >= THRESH_LOCK_MS))
        s_threshLocked = false;

    float threshLive = SEIZ_D * windowVariance(s_longBuf, s_longCount, LONG_WINDOW);
    float threshUse = s_threshLocked ? s_frozenThresh : threshLive;
    float S = windowVariance(s_shortBuf, s_shortCount, SHORT_WINDOW);
    *S_out = S;
    *th_out = threshUse;

    if (S > threshUse) {
        if (!s_threshLocked) {
            s_frozenThresh = threshLive;
            s_threshLocked = true;
            s_lockStartMs = millis();
        }
        if (s_therapyActive)
            return;
        s_seizurePointCount++;
        if (s_seizurePointCount >= SEIZ_T)
            startTherapy();
    }
}

static void processOneSample(int32_t raw) {
    float mv = convertToMilliV(raw);
    float vbp = s_bp.process(mv);
    ringPush(vbp);

    if (millis() - s_seizureWindowStartMs >= 1000u) {
        s_seizurePointCount = 0;
        s_seizureWindowStartMs = millis();
    }

    float S = 0, th = 0;
    runSeizureStep(&S, &th);

    s_printDiv++;
    if ((millis() - s_lastPrintMs) >= 100u) {
        s_lastPrintMs = millis();
        Serial.print(millis());
        Serial.print(',');
        Serial.print(raw);
        Serial.print(',');
        Serial.print(mv, 4);
        Serial.print(',');
        Serial.print(vbp, 4);
        Serial.print(',');
        Serial.print(S, 6);
        Serial.print(',');
        Serial.print(th, 6);
        Serial.print(',');
        Serial.println(s_therapyActive ? 1 : 0);
    }
}

static void runTherapySession(void) {
    if (!s_therapyActive)
        return;
    if (millis() - s_therapyStartMs >= s_therapyDurationMs) {
        s_therapyActive = false;
        s_seizurePointCount = 0;
        Serial.println(F("THERAPY_END"));
    }
}

void drdyInterruptHndlr() {
    drdyIntrFlag = true;
}

void setup() {
    Serial.begin(115200);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0 < 3000u))
        delay(10);

    s_bp.init(1000.0f, BP_F0_HZ, BP_Q);
    s_shortIdx = s_longIdx = 0;
    s_shortCount = s_longCount = 0;
    s_threshLocked = false;
    s_seizureWindowStartMs = millis();
    s_seizurePointCount = 0;
    s_therapyActive = false;
    s_printDiv = 0;
    s_lastPrintMs = 0;

    pinMode(ADS1220_CS_PIN, OUTPUT);
    digitalWrite(ADS1220_CS_PIN, HIGH);

    pc_ads1220.begin(ADS1220_CS_PIN, ADS1220_DRDY_PIN);
    pinMode(ADS1220_DRDY_PIN, INPUT_PULLUP);

    pc_ads1220.set_pga_gain(PGA_GAIN_1);
    pc_ads1220.set_data_rate(DR_1000SPS);
    pc_ads1220.select_mux_channels(MUX_AIN0_AIN1);
    pc_ads1220.set_conv_mode_continuous();
    attachInterrupt(digitalPinToInterrupt(ADS1220_DRDY_PIN), drdyInterruptHndlr, FALLING);
    pc_ads1220.Start_Conv();

    Serial.println(F("t_ms,raw,mV,filt_mV,S,thresh,therapy"));
}

void loop() {
    if (drdyIntrFlag) {
        drdyIntrFlag = false;
        if (digitalRead(ADS1220_DRDY_PIN) == LOW) {
            pc_ads1220.Read_Data();
            processOneSample(pc_ads1220.DataToInt());
        }
    }

    runTherapySession();
}
