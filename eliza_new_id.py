class New_ID:

    fields = [
        'type', 'subtype', 'bedroomCount', 'bathroomCount', 'province', 'locality', 'postCode',
        'habitableSurface', 'roomCount', 'hasAttic', 'hasBasement', 'hasDressingRoom',
        'diningRoomSurface', 'hasDiningRoom', 'buildingCondition', 'buildingConstructionYear',
        'facedeCount', 'floorCount', 'streetFacadeWidth', 'hasLift', 'floodZoneType',
        'heatingType', 'hasHeatPump', 'hasPhotovoltaicPanels', 'hasThermicPanels',
        'kitchenSurface', 'kitchenType', 'landSurface', 'hasLivingRoom', 'livingRoomSurface',
        'hasGarden', 'gardenSurface', 'gardenOrientation', 'parkingCountIndoor',
        'parkingCountOutdoor', 'hasAirConditioning', 'hasArmoredDoor', 'hasVisiophone',
        'hasOffice', 'toiletCount', 'hasSwimmingPool', 'hasFireplace', 'hasTerrace',
        'terraceSurface', 'terraceOrientation', 'epcScore', 'longitude', 'latitude',
        'cadastralIncome', 'primaryEnergyConsumptionPerSqm',
    ]

    defaults = {
        # Strings
        'type': -1,
        'subtype': -1,
        'floodZoneType': 0,
        'heatingType': -1,
        'kitchenType': -1,
        'gardenOrientation': -1,
        'terraceOrientation': -1,
        'epcScore': -1,

        # Booleans
        'hasAttic': 0,
        'hasBasement': 0,
        'hasDressingRoom': 0,
        'hasDiningRoom': 0,
        'hasLift': 0,
        'hasHeatPump': 0,
        'hasPhotovoltaicPanels': 0,
        'hasThermicPanels': 0,
        'hasLivingRoom': 0,
        'hasGarden': 0,
        'hasAirConditioning': 0,
        'hasArmoredDoor': 0,
        'hasVisiophone': 0,
        'hasOffice': 0,
        'hasSwimmingPool': 0,
        'hasFireplace': 0,
        'hasTerrace': 0,

        # Integers
        'bedroomCount': 0,
        'bathroomCount': 0,
        'province': 0,
        'locality': 0,
        'postCode': 0,
        'roomCount': 0,
        'diningRoomSurface': 0,
        'buildingCondition': 0,
        'buildingConstructionYear': 0,
        'facedeCount': 0,
        'floorCount': 0,
        'streetFacadeWidth': 0,
        'parkingCountIndoor': 0,
        'parkingCountOutdoor': 0,
        'toiletCount': 0,
        'terraceSurface': 0,
        
        # Floats
        'habitableSurface': -1,
        'kitchenSurface': -1,
        'landSurface': -1,
        'livingRoomSurface': -1,
        'gardenSurface': -1,
        'longitude': 0.0,
        'latitude': 0.0,
        'cadastralIncome': -1,
        'primaryEnergyConsumptionPerSqm': -1
    }

    def __init__(self, **kwargs):
        for field in self.fields:
            value = kwargs.get(field, self.defaults.get(field))
            setattr(self, field, value)
            # self.field = value


    def to_list(self) -> list:
        """
        Retourne les valeurs dans l'ordre défini par 'fields', prêt pour le modèle.
        """
        return [getattr(self, field) for field in self.fields]
