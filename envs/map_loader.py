import json
import numpy as np
from typing import Dict, List, Any

class MapLoader:
    """Loads and validates map configurations from JSON files"""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Main entry point - loads and processes map config"""
        with open(config_path) as f:
            raw_config = json.load(f)
        
        MapLoader._validate_config(raw_config)
        
        return {
            "lanes": MapLoader._process_lanes(raw_config["lanes"]),
            "intersections": MapLoader._process_intersections(
                raw_config.get("intersections", [])
            ),
            "traffic_lights": MapLoader._process_traffic_lights(
                raw_config.get("traffic_lights", [])
            ),
            "metadata": raw_config.get("metadata", {})
        }

    @staticmethod
    def _validate_config(config: Dict) -> None:
        """Ensure required fields are present"""
        if "lanes" not in config:
            raise ValueError("Map config must contain 'lanes' section")
        
        required_lane_fields = {"id", "start", "end", "width"}
        for lane in config["lanes"]:
            if not required_lane_fields.issubset(lane.keys()):
                missing = required_lane_fields - set(lane.keys())
                raise ValueError(f"Lane {lane.get('id', '?')} missing fields: {missing}")

    @staticmethod
    def _process_lanes(raw_lanes: List[Dict]) -> List[Dict]:
        """Convert raw lane data to environment-ready format"""
        processed = []
        for lane in raw_lanes:
            processed.append({
                "id": lane["id"],
                "start": np.array(lane["start"], dtype=np.float32),
                "end": np.array(lane["end"], dtype=np.float32),
                "width": float(lane["width"]),
                "direction": MapLoader._calculate_lane_direction(
                    lane["start"], lane["end"]
                )
            })
        return processed

    @staticmethod
    def _calculate_lane_direction(start: List[float], end: List[float]) -> np.ndarray:
        """Calculate normalized direction vector for lane"""
        vec = np.array(end) - np.array(start)
        return vec / np.linalg.norm(vec)

    @staticmethod
    def _process_intersections(raw_intersections: List[Dict]) -> List[Dict]:
        """Process intersection polygons"""
        processed = []
        for ix in raw_intersections:
            processed.append({
                "id": ix["id"],
                "vertices": np.array(ix["vertices"], dtype=np.float32),
                "type": ix.get("type", "crossing")
            })
        return processed

    @staticmethod
    def _process_traffic_lights(raw_tls: List[Dict]) -> List[Dict]:
        """Process traffic light data"""
        processed = []
        for tl in raw_tls:
            processed.append({
                "id": tl["id"],
                "position": np.array(tl["position"], dtype=np.float32),
                "orientation": float(tl["orientation"]),  # radians
                "cycle": tl.get("cycle", [30, 3, 30]),  # [green, yellow, red] durations
                "state": tl.get("initial_state", "red")
            })
        return processed
