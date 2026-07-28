from openmc_agent.executor import _region_expression_to_python


def test_region_expression_accepts_input_derived_decimal_surface_ids() -> None:
    expression = (
        "+surf_u_fuel_region_1_2.11pct_fuel_inner "
        "-surf_u_fuel_region_1_2.11pct_fuel_outer"
    )

    rendered = _region_expression_to_python(expression)

    assert "(+surfaces['surf_u_fuel_region_1_2.11pct_fuel_inner'])" in rendered
    assert "(-surfaces['surf_u_fuel_region_1_2.11pct_fuel_outer'])" in rendered


def test_region_expression_accepts_numeric_leading_surface_ids() -> None:
    assert _region_expression_to_python("-11pct_fuel_inner") == (
        "(-surfaces['11pct_fuel_inner'])"
    )
