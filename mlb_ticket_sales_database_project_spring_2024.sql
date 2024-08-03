
CREATE DATABASE IF NOT EXISTS `henry_project` /*!40100 DEFAULT CHARACTER SET
utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `henry_project`;
-- MySQL dump 10.13 Distrib 8.0.36, for Win64 (x86_64)
--
-- Host: localhost Database: henry_project
-- ------------------------------------------------------
-- Server version 8.3.0
/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
--
-- Table structure for table `customer`
--
DROP TABLE IF EXISTS `customer`;
/*!40101 SET @saved_cs_client = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `customer` (
`customer_id` int NOT NULL,
`ticket_id` int DEFAULT NULL,
`name` varchar(100) DEFAULT NULL,
`email` varchar(255) DEFAULT NULL,
`phone_number` varchar(20) DEFAULT NULL,
PRIMARY KEY (`customer_id`),
KEY `ticket_id` (`ticket_id`),
CONSTRAINT `customer_ibfk_1` FOREIGN KEY (`ticket_id`) REFERENCES `ticket`
(`ticket_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
--
-- Dumping data for table `customer`
--
LOCK TABLES `customer` WRITE;
/*!40000 ALTER TABLE `customer` DISABLE KEYS */;
INSERT INTO `customer` VALUES (176,743,'Alex
Johnson','alexjohnson@gmail.com','+1234567890'),(248,975,'Sarah
Smith','sarahsmith@outlook.com','+1987654321'),(359,385,'Ryan
Davis','ryandavis@yahoo.com','+1122334455'),(412,137,'Emily
Brown','emilybrown@yahoo.com','+1567890123'),(597,609,'Jacob
Martinez','jacobmartinez@gmail.com','+1654321890');
/*!40000 ALTER TABLE `customer` ENABLE KEYS */;
UNLOCK TABLES;
--
-- Table structure for table `game`
--
DROP TABLE IF EXISTS `game`;
/*!40101 SET @saved_cs_client = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `game` (
`game_id` int NOT NULL,
`stadium_id` int DEFAULT NULL,
`date` date DEFAULT NULL,
`time` time DEFAULT NULL,
`teams` varchar(225) DEFAULT NULL,
PRIMARY KEY (`game_id`),
KEY `stadium_id` (`stadium_id`),
CONSTRAINT `game_ibfk_1` FOREIGN KEY (`stadium_id`) REFERENCES `stadium`
(`stadium_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
--
-- Dumping data for table `game`
--
LOCK TABLES `game` WRITE;
/*!40000 ALTER TABLE `game` DISABLE KEYS */;
INSERT INTO `game` VALUES (98,514,'2024-05-06','20:10:00','Los Angeles Dodgers vs.
Arizona Diamondbacks'),(173,237,'2024-04-15','19:05:00','New York Yankees vs.
Boston Red Sox'),(224,358,'2024-04-18','19:10:00','Boston Red Sox vs. New York
Yankees'),(352,691,'2024-05-07','13:20:00','Chicago Cubs vs. Milwaukee Brewers'),
(375,789,'2024-05-01','19:10:00','Houston Astros vs. Texas Rangers'),
(406,514,'2024-04-16','20:10:00','Los Angeles Dodgers vs. San Francisco Giants'),
(434,846,'2024-05-04','20:10:00','Seattle Mariners vs. Oakland Athletics'),
(557,972,'2024-04-19','18:45:00','San Francisco Giants vs. Los Angeles Dodgers'),
(675,358,'2024-05-08','19:10:00','Boston Red Sox vs. Baltimore Orioles'),
(688,163,'2024-05-02','16:05:00','Washington Nationals vs. New York Mets'),
(739,691,'2024-04-17','13:20:00','Chicago Cubs vs. St. Louis Cardinals'),
(765,237,'2024-05-05','19:05:00','New York Yankees vs. Toronto Blue Jays'),
(890,425,'2024-04-30','18:35:00','Los Angeles Angels vs. Houston Astros'),
(901,572,'2024-05-03','13:40:00','San Diego Padres vs. Los Angeles Dodgers'),
(922,972,'2024-05-09','18:45:00','San Francisco Giants vs. Colorado Rockies');
/*!40000 ALTER TABLE `game` ENABLE KEYS */;
UNLOCK TABLES;
--
-- Table structure for table `marketingcampaign`
--
DROP TABLE IF EXISTS `marketingcampaign`;
/*!40101 SET @saved_cs_client = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `marketingcampaign` (
`campaign_id` int NOT NULL,
`description` text,
`start_date` date DEFAULT NULL,
`end_date` date DEFAULT NULL,
PRIMARY KEY (`campaign_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
--
-- Dumping data for table `marketingcampaign`
--
LOCK TABLES `marketingcampaign` WRITE;
/*!40000 ALTER TABLE `marketingcampaign` DISABLE KEYS */;
INSERT INTO `marketingcampaign` VALUES (127,'End of Season Sale - Win free
tickets!','2024-09-01','2024-09-30'),(498,'Opening Day Special - 20% off on all
tickets!','2024-04-01','2024-04-15'),(513,'Midseason Madness - Buy one, get one
free!','2024-06-01','2024-06-30');
/*!40000 ALTER TABLE `marketingcampaign` ENABLE KEYS */;
UNLOCK TABLES;
--
-- Table structure for table `marketingteam`
--
DROP TABLE IF EXISTS `marketingteam`;
/*!40101 SET @saved_cs_client = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `marketingteam` (
`marketer_id` int NOT NULL,
`name` varchar(100) DEFAULT NULL,
`email` varchar(255) DEFAULT NULL,
`phone_number` varchar(20) DEFAULT NULL,
`role` varchar(50) DEFAULT NULL,
PRIMARY KEY (`marketer_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
--
-- Dumping data for table `marketingteam`
--
LOCK TABLES `marketingteam` WRITE;
/*!40000 ALTER TABLE `marketingteam` DISABLE KEYS */;
INSERT INTO `marketingteam` VALUES (117,'Olivia
Garcia','oliviagarcia@mlb.com','+1122334455','Marketing Specialist'),(129,'Matthew
Taylor','matthewtaylor@mlb.com','+1234567890','Marketing Manager'),(146,'Daniel
Soto','danielsoto@mlb.com','+1567890123','Marketing Coordinator');
/*!40000 ALTER TABLE `marketingteam` ENABLE KEYS */;
UNLOCK TABLES;
--
-- Table structure for table `salesteam`
--
DROP TABLE IF EXISTS `salesteam`;
/*!40101 SET @saved_cs_client = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `salesteam` (
`salesperson_id` int NOT NULL,
`name` varchar(100) DEFAULT NULL,
`email` varchar(255) DEFAULT NULL,
`phone_number` varchar(20) DEFAULT NULL,
`role` varchar(50) DEFAULT NULL,
PRIMARY KEY (`salesperson_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
--
-- Dumping data for table `salesteam`
--
LOCK TABLES `salesteam` WRITE;
/*!40000 ALTER TABLE `salesteam` DISABLE KEYS */;
INSERT INTO `salesteam` VALUES (110,'Jessica
Wilson','jessicawilson@mlb.com','+1987654321','Sales Representative'),(119,'David
Rodriguez','davidrodriguez@mlb.com','+1654321890','Sales Associate'),(151,'Michael
Adams','michaeladams@mlb.com','+1123456789','Sales Manager');
/*!40000 ALTER TABLE `salesteam` ENABLE KEYS */;
UNLOCK TABLES;
--
-- Table structure for table `stadium`
--
DROP TABLE IF EXISTS `stadium`;
/*!40101 SET @saved_cs_client = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `stadium` (
`stadium_id` int NOT NULL,
`name` varchar(100) DEFAULT NULL,
`capacity` int DEFAULT NULL,
`location` varchar(255) DEFAULT NULL,
PRIMARY KEY (`stadium_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
--
-- Dumping data for table `stadium`
--
LOCK TABLES `stadium` WRITE;
/*!40000 ALTER TABLE `stadium` DISABLE KEYS */;
INSERT INTO `stadium` VALUES (163,'Nationals Park',41339,'Washington, D.C.'),
(237,'Yankee Stadium',47309,'New York City, New York'),(358,'Fenway
Park',37755,'Boston, Massachusetts'),(425,'Angel Stadium of
Anaheim',45423,'Anaheim, California'),(514,'Dodger Stadium',56000,'Los Angeles,
California'),(572,'Petco Park',40209,'San Diego, California'),(691,'Wrigley
Field',41649,'Chicago, Illinois'),(789,'Minute Maid Park',41168,'Houston, Texas'),
(846,'T-Mobile Park',47744,'Seattle, Washington'),(972,'Oracle Park',41915,'San
Francisco, California');
/*!40000 ALTER TABLE `stadium` ENABLE KEYS */;
UNLOCK TABLES;
--
-- Table structure for table `ticket`
--
DROP TABLE IF EXISTS `ticket`;
/*!40101 SET @saved_cs_client = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ticket` (
`ticket_id` int NOT NULL,
`game_id` int DEFAULT NULL,
`stadium_id` int DEFAULT NULL,
`seat_section` varchar(20) DEFAULT NULL,
`seat_row` varchar(20) DEFAULT NULL,
`seat_number` varchar(20) DEFAULT NULL,
`price` decimal(10,2) DEFAULT NULL,
`status` varchar(20) DEFAULT NULL,
PRIMARY KEY (`ticket_id`),
KEY `game_id` (`game_id`),
KEY `stadium_id` (`stadium_id`),
CONSTRAINT `ticket_ibfk_1` FOREIGN KEY (`game_id`) REFERENCES `game` (`game_id`),
CONSTRAINT `ticket_ibfk_2` FOREIGN KEY (`stadium_id`) REFERENCES `stadium`
(`stadium_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
--
-- Dumping data for table `ticket`
--
LOCK TABLES `ticket` WRITE;
/*!40000 ALTER TABLE `ticket` DISABLE KEYS */;
INSERT INTO `ticket` VALUES (137,688,163,'113','7','12',100.00,'Sold'),
(264,406,514,'211','12','4',75.00,'Available'),
(385,765,237,'216','9','3',50.00,'Sold'),
(492,675,358,'324','5','4',120.00,'Available'),
(571,352,691,'102','16','9',90.00,'Sold'),
(609,922,972,'115','8','15',120.00,'Sold'),
(743,765,237,'203','10','8',85.00,'Sold'),
(818,375,789,'315','6','12',65.00,'Available'),
(936,434,846,'331','4','21',150.00,'Available'),
(975,890,425,'108','18','5',110.00,'Sold');
/*!40000 ALTER TABLE `ticket` ENABLE KEYS */;
UNLOCK TABLES;
--
-- Table structure for table `tickettransaction`
--
DROP TABLE IF EXISTS `tickettransaction`;
/*!40101 SET @saved_cs_client = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `tickettransaction` (
`transaction_id` int NOT NULL,
`ticket_id` int DEFAULT NULL,
`salesperson_id` int DEFAULT NULL,
`customer_id` int DEFAULT NULL,
`transaction_date` date DEFAULT NULL,
`amount` decimal(10,2) DEFAULT NULL,
`payment_method` varchar(50) DEFAULT NULL,
PRIMARY KEY (`transaction_id`),
KEY `ticket_id` (`ticket_id`),
KEY `salesperson_id` (`salesperson_id`),
KEY `customer_id` (`customer_id`),
CONSTRAINT `tickettransaction_ibfk_1` FOREIGN KEY (`ticket_id`) REFERENCES
`ticket` (`ticket_id`),
CONSTRAINT `tickettransaction_ibfk_2` FOREIGN KEY (`salesperson_id`) REFERENCES
`salesteam` (`salesperson_id`),
CONSTRAINT `tickettransaction_ibfk_3` FOREIGN KEY (`customer_id`) REFERENCES
`customer` (`customer_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
--
-- Dumping data for table `tickettransaction`
--
LOCK TABLES `tickettransaction` WRITE;
/*!40000 ALTER TABLE `tickettransaction` DISABLE KEYS */;
INSERT INTO `tickettransaction` VALUES (182,609,110,597,'2024-04-14',120.00,'Credit
Card'),(273,137,119,412,'2024-04-12',100.00,'PayPal'),(396,743,119,176,'2024-04-
10',85.00,'Cash'),(481,385,151,359,'2024-04-13',50.00,'Credit Card'),
(597,975,110,248,'2024-04-12',110.00,'PayPal');
/*!40000 ALTER TABLE `tickettransaction` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;
/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
-- Dump completed on 2024-04-14 22:25:24
