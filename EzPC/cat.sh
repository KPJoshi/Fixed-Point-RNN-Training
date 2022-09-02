#! /usr/bin/env bash

M="manual"
A="autogen"
E=".ezpc"

cat $M/constants$E $A/piecewise$E $M/fxplib$E $M/conv$E $A/testFunc$E $A/trainFunc$E $M/mainA$E $A/init$E $M/mainB$E $A/trainCall$E $M/mainC$E $A/testCall$E $M/mainD$E > build/lstm$E
