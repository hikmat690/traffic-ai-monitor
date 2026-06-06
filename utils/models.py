from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
engine = create_engine("sqlite:///data/traffic.db", echo=False)
Session = sessionmaker(bind=engine)


class VehicleCount(Base):
    __tablename__ = "vehicle_counts"
    id        = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    zone_name = Column(String)
    count     = Column(Integer)
    vehicle_type = Column(String)  # car / truck / motorcycle / bus


class ParkingSnapshot(Base):
    __tablename__ = "parking_snapshots"
    id           = Column(Integer, primary_key=True)
    timestamp    = Column(DateTime, default=datetime.utcnow)
    zone_name    = Column(String)
    total_slots  = Column(Integer)
    occupied     = Column(Integer)
    occupancy_pct = Column(Float)


class AlertLog(Base):
    __tablename__ = "alert_log"
    id        = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    zone_name = Column(String)
    message   = Column(String)
    sent_sms  = Column(Boolean, default=False)


def init_db():
    import os
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(engine)
