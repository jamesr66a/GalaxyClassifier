ls spiral/scaled | grep jpg | sed 's/$/ 0/' | sed 's/^/.\/images\/spiral\/scaled\//' > spiral/scaled/example_map
ls lenticular/scaled | grep jpg | sed 's/$/ 1/' | sed 's/^/.\/images\/lenticular\/scaled\//' > lenticular/scaled/example_map
ls irregular/scaled | grep jpg | sed 's/$/ 2/' | sed 's/^/.\/images\/irregular\/scaled\//' > irregular/scaled/example_map
ls elliptical/scaled | grep jpg | sed 's/$/ 3/' | sed 's/^/.\/images\/elliptical\/scaled\//' > elliptical/scaled/example_map
