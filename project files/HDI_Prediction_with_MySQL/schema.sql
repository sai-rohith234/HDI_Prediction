-- MySQL schema for the Human Development Index (HDI) Prediction System.
-- Run once with: mysql -u root -p < schema.sql

CREATE DATABASE IF NOT EXISTS hdi_prediction_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE hdi_prediction_db;

CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(120) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS country (
    country_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    country_name VARCHAR(120) NOT NULL UNIQUE,
    iso_code CHAR(3) NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS dataset (
    dataset_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    dataset_name VARCHAR(200) NOT NULL UNIQUE,
    source_path VARCHAR(500) NOT NULL,
    description TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ml_model (
    model_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    dataset_id BIGINT UNSIGNED NOT NULL,
    model_name VARCHAR(160) NOT NULL UNIQUE,
    algorithm VARCHAR(100) NOT NULL,
    model_path VARCHAR(500) NOT NULL,
    trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    CONSTRAINT fk_model_dataset
        FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS user_session (
    session_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL,
    session_token CHAR(36) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NULL,
    CONSTRAINT fk_session_user
        FOREIGN KEY (user_id) REFERENCES users(user_id)
        ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS hdi_input_data (
    input_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL,
    country_id BIGINT UNSIGNED NOT NULL,
    life_expectancy DECIMAL(5,2) NOT NULL,
    mean_years_of_schooling DECIMAL(5,2) NOT NULL,
    expected_years_of_schooling DECIMAL(5,2) NOT NULL,
    gni_per_capita DECIMAL(14,2) NOT NULL,
    submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_input_user
        FOREIGN KEY (user_id) REFERENCES users(user_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_input_country
        FOREIGN KEY (country_id) REFERENCES country(country_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    INDEX idx_input_user_submitted (user_id, submitted_at),
    INDEX idx_input_country (country_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS hdi_prediction (
    prediction_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    input_id BIGINT UNSIGNED NOT NULL UNIQUE,
    model_id BIGINT UNSIGNED NOT NULL,
    predicted_hdi DECIMAL(6,5) NOT NULL,
    predicted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_predicted_hdi CHECK (predicted_hdi >= 0 AND predicted_hdi <= 1),
    CONSTRAINT fk_prediction_input
        FOREIGN KEY (input_id) REFERENCES hdi_input_data(input_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_prediction_model
        FOREIGN KEY (model_id) REFERENCES ml_model(model_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    INDEX idx_prediction_model_time (model_id, predicted_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS visualization_report (
    report_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    prediction_id BIGINT UNSIGNED NOT NULL,
    report_type VARCHAR(100) NOT NULL,
    report_path VARCHAR(500) NULL,
    report_payload JSON NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_report_prediction
        FOREIGN KEY (prediction_id) REFERENCES hdi_prediction(prediction_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    INDEX idx_report_prediction (prediction_id)
) ENGINE=InnoDB;
