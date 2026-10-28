"""Civilian building pools shared by the regions (all RA2 objects; availability per theater is checked by the engine)."""
POOLS = {
    'highrise': ['CANEWY01', 'CANEWY06', 'CANEWY07', 'CANEWY08', 'CANEWY13', 'CANEWY14', 'CANEWY15', 'CANWY05',
                 'CANWY22', 'CANWY24', 'CANWY25', 'CACHIG01', 'CACHIG02', 'CACHIG03', 'CAMIAM06', 'CAMIAM07',
                 'CAWASH04', 'CAWASH06'],
    'houses': ['CAHSE01', 'CAHSE02', 'CAHSE03', 'CAHSE04', 'CAHSE05', 'CAHSE06', 'CAHSE07', 'CAEUR1', 'CAEUR2'],
    'suburb': ['CAHSE01', 'CAHSE02', 'CAHSE04', 'CAGAS01', 'CAMSC10', 'CAMIAM01', 'CAMIAM02', 'CANEWY16',
               'CANEWY17', 'CANEWY18', 'CASWST01'],
    'industrial': ['CAIND01', 'CATS01', 'CACHIG06', 'CANWY09', 'CANEWY12', 'CABUNK01'],
    'farm': ['CAFARM01', 'CABARN02', 'CAFRMA', 'CAFARM02', 'CATS01'],
    'village': ['CAEUR1', 'CAEUR2', 'CAFRMA', 'CABARN02', 'CAHSE05', 'CAHSE06'],
}
POOLS['fill'] = POOLS['houses'] + POOLS['suburb']
POOLS['small'] = ['CAHSE05', 'CAHSE06', 'CAEUR1', 'CAEUR2', 'CABARN02', 'CAFRMA', 'CAHSE01', 'CAHSE04', 'CAFARM02']
