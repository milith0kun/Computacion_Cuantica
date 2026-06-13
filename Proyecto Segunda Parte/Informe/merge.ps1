$anteproyecto = 'D:\Proyectos\Cuantica\Proyecto Semestral\Anteproyecto\main.tex'
$cap2 = 'd:\Proyectos\Cuantica\Proyecto Segunda Parte\Informe\contenido\cap_2_marco_teorico.tex'

$lines_ante = Get-Content $anteproyecto -Encoding UTF8
$teoria = $lines_ante[253..505]

$lines_cap2 = Get-Content $cap2 -Encoding UTF8
$header = $lines_cap2[0..4]
$footer = $lines_cap2[5..($lines_cap2.Length - 1)]

$new_content = $header + $teoria + $footer
$new_content | Set-Content $cap2 -Encoding UTF8
