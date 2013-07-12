//Bessere Implementierung mit C
typedef struct {
    float x_old, y_old, z_old;
    float x,y,z;
    float vx,vy,vz;
    float ax,ay,az;
    float mass;
} Planet;

__kernel void leap_frog(__global Planet* universe, float dt, int planets, float maxTime) {
    int i = get_global_id(0);
    
    for (float t=0; t<maxTime; t += dt) {
        universe[i].x = universe[i].x_old + universe[i].vx*dt;
        universe[i].y = universe[i].y_old + universe[i].vy*dt;
        universe[i].z = universe[i].z_old + universe[i].vz*dt;
        
        //Berechnen der neuen Beschleunigungen bei r(t)
        float rij2, rijx, rijy, rijz, rij;
        float gamma = 1;
        universe[i].ax = 0;
        universe[i].ay = 0;
        universe[i].az = 0;
        
        for (int j = 0; j<planets; j++) {
            if (i == j) continue;
            
            rijx = universe[j].x - universe[i].x;
            rijy = universe[j].y - universe[i].y;
            rijz = universe[j].z - universe[i].z;
            
            rij2 = rijx*rijx+rijy*rijy+rijz*rijz;
            rij  = sqrt(rij2);
            
            universe[i].ax += (rijx/rij)*gamma*universe[j].mass/rij2;
            universe[i].ay += (rijy/rij)*gamma*universe[j].mass/rij2;
            universe[i].az += (rijz/rij)*gamma*universe[j].mass/rij2;
            
            //          Newton's 3. Gesetz
            //          universe[j].ax =- (rijx/rij)*gamma*universe[j].mass/rij2;
            //          universe[j].ay =- (rijy/rij)*gamma*universe[j].mass/rij2;
            //          universe[j].az =- (rijz/rij)*gamma*universe[j].mass/rij2;
        }
        
        
        //Berechnen der neuen Geschwindigkeit bei t+0.5*dt
        universe[i].vx += universe[i].ax*dt;
        universe[i].vy += universe[i].ay*dt;
        universe[i].vz += universe[i].az*dt;
    }
}