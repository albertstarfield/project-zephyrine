	.arch armv8.5-a
	.build_version macos,  16, 0
	.text
Ltext0:
	.file 1 "/Users/albertstarfield/Documents/misc/AI/project-zephyrine/systemCore/engineMain/_excludefromRuntime_reverseEngineeringAssets/ada_courseRE/hello_sekai_spark/src/message_generator.adb"
	.align	2
	.globl _message_generator__get_message
_message_generator__get_message:
LFB2:
	.loc 1 5 4
	stp	x29, x30, [sp, -16]!
LCFI0:
	mov	x29, sp
LCFI1:
	.loc 1 9 7
	mov	x1, 4
	mov	x0, 44
	bl	_system__secondary_stack__ss_allocate
LVL0:
	mov	x1, x0
	.loc 1 9 7 is_stmt 0 discriminator 1
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	ldr	q29, [x0]
	ldr	q30, [x0, 16]
	ldr	q31, [x0, 28]
	str	q29, [x1]
	str	q30, [x1, 16]
	str	q31, [x1, 28]
	.loc 1 10 8 is_stmt 1
	add	x0, x1, 8
	ldp	x29, x30, [sp], 16
LCFI2:
	ret
LFE2:
	.const
	.align	2
lC0:
	.word	1
	.word	36
	.ascii "Hello, Sekai! (Proven and Separated)"
	.text
	.globl _message_generator_E
	.data
	.align	1
_message_generator_E:
	.space 2
	.section __DWARF,__debug_frame,regular,debug
Lsection__debug_frame:
Lframe0:
	.set L$set$0,LECIE0-LSCIE0
	.long L$set$0
LSCIE0:
	.long	0xffffffff
	.byte	0x3
	.ascii "\0"
	.uleb128 0x1
	.sleb128 -8
	.uleb128 0x1e
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LECIE0:
LSFDE0:
	.set L$set$1,LEFDE0-LASFDE0
	.long L$set$1
LASFDE0:
	.set L$set$2,Lframe0-Lsection__debug_frame
	.long L$set$2
	.quad	LFB2
	.set L$set$3,LFE2-LFB2
	.quad L$set$3
	.byte	0x4
	.set L$set$4,LCFI0-LFB2
	.long L$set$4
	.byte	0xe
	.uleb128 0x10
	.byte	0x9d
	.uleb128 0x2
	.byte	0x9e
	.uleb128 0x1
	.byte	0x4
	.set L$set$5,LCFI1-LCFI0
	.long L$set$5
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$6,LCFI2-LCFI1
	.long L$set$6
	.byte	0xde
	.byte	0xdd
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE0:
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$7,LECIE1-LSCIE1
	.long L$set$7
LSCIE1:
	.long	0
	.byte	0x3
	.ascii "zR\0"
	.uleb128 0x1
	.sleb128 -8
	.uleb128 0x1e
	.uleb128 0x1
	.byte	0x10
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LECIE1:
LSFDE3:
	.set L$set$8,LEFDE3-LASFDE3
	.long L$set$8
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB2-.
	.set L$set$9,LFE2-LFB2
	.quad L$set$9
	.uleb128 0
	.byte	0x4
	.set L$set$10,LCFI0-LFB2
	.long L$set$10
	.byte	0xe
	.uleb128 0x10
	.byte	0x9d
	.uleb128 0x2
	.byte	0x9e
	.uleb128 0x1
	.byte	0x4
	.set L$set$11,LCFI1-LCFI0
	.long L$set$11
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$12,LCFI2-LCFI1
	.long L$set$12
	.byte	0xde
	.byte	0xdd
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE3:
	.text
Letext0:
	.file 2 "<built-in>"
	.section __DWARF,__debug_info,regular,debug
Lsection__debug_info:
Ldebug_info0:
	.long	0x519
	.short	0x4
	.set L$set$13,Ldebug_abbrev0-Lsection__debug_abbrev
	.long L$set$13
	.byte	0x8
	.uleb128 0x1
	.ascii "GNU Ada 15.0.1 20250418 (prerelease) -Og -gnatA -ffunction-sections -fdata-sections -g -gnatwa -gnatw.X -gnatVa -gnaty3 -gnatya -gnatyA -gnatyB -gnatyb -gnatyc -gnaty-d -gnatye -gnatyf -gnatyh -gnatyi -gnatyI -gnatyk -gnatyl -gnatym -gnatyn -gnatyO -gnatyp -gnatyr -gnatyS -gnatyt -gnatyu -gnatyx -gnatW8 -gnatR2js -gnatws -gnatis -gnatec=/private/var/folders/vj/2td27x090rqc1ln_jr_6v83m0000gn/T/GPR.27480/GNAT-TEMP-000003.TMP -gnatem=/private/var/folders/vj/2td27x090rqc1ln_jr_6v83m0000gn/T/GPR.27480/GNAT-TEMP-000004.TMP -mmacosx-version-min=16.0.0 -mcpu=apple-m1 -mlittle-endian -mabi=lp64 -fPIC\0"
	.byte	0xd
	.ascii "/Users/albertstarfield/Documents/misc/AI/project-zephyrine/systemCore/engineMain/_excludefromRuntime_reverseEngineeringAssets/ada_courseRE/hello_sekai_spark/src/message_generator.adb\0"
	.ascii "/Users/albertstarfield/Documents/misc/AI/project-zephyrine/systemCore/engineMain/_excludefromRuntime_reverseEngineeringAssets/ada_courseRE/hello_sekai_spark/obj/development/gnatprove/data_representation\0"
	.quad	Ltext0
	.set L$set$14,Letext0-Ltext0
	.quad L$set$14
	.set L$set$15,Ldebug_line0-Lsection__debug_line
	.long L$set$15
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.ascii "system__address\0"
	.uleb128 0x3
	.byte	0x8
	.byte	0x5
	.ascii "system__storage_elements__Tstorage_offsetB\0"
	.uleb128 0x4
	.ascii "string\0"
	.byte	0x10
	.byte	0x2
	.byte	0
	.long	0x493
	.uleb128 0x5
	.ascii "P_ARRAY\0"
	.byte	0x2
	.byte	0
	.long	0x45a
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x493
	.uleb128 0x7
	.byte	0x8
	.byte	0x2
	.byte	0
	.long	0x481
	.uleb128 0x5
	.ascii "LB0\0"
	.byte	0x2
	.byte	0
	.long	0x4ca
	.byte	0
	.uleb128 0x5
	.ascii "UB0\0"
	.byte	0x2
	.byte	0
	.long	0x4ca
	.byte	0x4
	.byte	0
	.uleb128 0x5
	.ascii "P_BOUNDS\0"
	.byte	0x2
	.byte	0
	.long	0x4dd
	.byte	0x8
	.byte	0
	.uleb128 0x8
	.long	0x4bd
	.long	0x4b2
	.uleb128 0x9
	.long	0x4b2
	.uleb128 0x6
	.byte	0x97
	.byte	0x23
	.uleb128 0x8
	.byte	0x6
	.byte	0x94
	.byte	0x4
	.uleb128 0x8
	.byte	0x97
	.byte	0x23
	.uleb128 0x8
	.byte	0x6
	.byte	0x23
	.uleb128 0x4
	.byte	0x94
	.byte	0x4
	.byte	0
	.uleb128 0x2
	.byte	0x4
	.byte	0x5
	.ascii "integer\0"
	.uleb128 0x2
	.byte	0x1
	.byte	0x8
	.ascii "character\0"
	.uleb128 0xa
	.sleb128 2147483647
	.ascii "positive\0"
	.long	0x4b2
	.uleb128 0x6
	.byte	0x8
	.long	0x460
	.uleb128 0xb
	.ascii "message_generator__get_message\0"
	.byte	0x1
	.byte	0x5
	.byte	0x4
	.long	0x43b
	.quad	LFB2
	.set L$set$16,LFE2-LFB2
	.quad L$set$16
	.uleb128 0x1
	.byte	0x9c
	.byte	0
	.section __DWARF,__debug_abbrev,regular,debug
Lsection__debug_abbrev:
Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0x8
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x1b
	.uleb128 0x8
	.uleb128 0x2134
	.uleb128 0x19
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x34
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x4
	.uleb128 0x13
	.byte	0x1
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x5
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x6
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x7
	.uleb128 0x13
	.byte	0x1
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x8
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x9
	.uleb128 0x21
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x22
	.uleb128 0x18
	.uleb128 0x2f
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0xa
	.uleb128 0x21
	.byte	0
	.uleb128 0x2f
	.uleb128 0xd
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xb
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.byte	0
	.byte	0
	.byte	0
	.section __DWARF,__debug_pubnames,regular,debug
Lsection__debug_pubnames:
	.long	0x31
	.short	0x2
	.set L$set$17,Ldebug_info0-Lsection__debug_info
	.long L$set$17
	.long	0x51d
	.long	0x4e3
	.ascii "message_generator__get_message\0"
	.long	0
	.section __DWARF,__debug_pubtypes,regular,debug
Lsection__debug_pubtypes:
	.long	0x47
	.short	0x2
	.set L$set$18,Ldebug_info0-Lsection__debug_info
	.long L$set$18
	.long	0x51d
	.long	0x3fa
	.ascii "system__address\0"
	.long	0x4b2
	.ascii "integer\0"
	.long	0x4bd
	.ascii "character\0"
	.long	0x43b
	.ascii "string\0"
	.long	0
	.section __DWARF,__debug_aranges,regular,debug
Lsection__debug_aranges:
	.long	0x2c
	.short	0x2
	.set L$set$19,Ldebug_info0-Lsection__debug_info
	.long L$set$19
	.byte	0x8
	.byte	0
	.short	0
	.short	0
	.quad	Ltext0
	.set L$set$20,Letext0-Ltext0
	.quad L$set$20
	.quad	0
	.quad	0
	.section __DWARF,__debug_line,regular,debug
Lsection__debug_line:
Ldebug_line0:
	.section __DWARF,__debug_str,regular,debug
Lsection__debug_str:
	.ident	"GCC: (GNU) 15.0.1 20250418 (prerelease)"
	.subsections_via_symbols
