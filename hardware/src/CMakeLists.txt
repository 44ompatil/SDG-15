idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "../include"
    REQUIRES
        driver
        esp_driver_gpio
        esp32-camera
        esp_wifi
        esp_event
        esp_netif
        esp_http_client
        esp_timer
        lwip
        nvs_flash
)

target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)
